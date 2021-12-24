#ifndef __APPROXIMATOR_H_
#define __APPROXIMATOR_H_

#include "Eigen/Dense"
#include <omp.h>
#include <vector>
#include <string>
#include <stdexcept>

/** 
 *  @brief Provides a generic interface for value function approximators. It is
 *         restricted to discrete action spaces.
 */
class Approximator {
  protected:
    std::vector<omp_lock_t> action_locks; //!< Lock for each action when using OMP

    /**
     * @brief Predicts the value of the given state-action pair.
     * 
     * @param state State vector
     * @param action Action value
     * @return State-action value 
     */
    virtual double predict_implementation(
        Eigen::Ref<const Eigen::VectorXd> state,
        int action) {
        throw std::logic_error("Not implemented");
    }

    /**
     * @brief Predicts the values of multiple state-action pairs.
     * 
     * @param state State vector
     * @param actions Vector containing multiple actions to evaluate.
     * @return State-action values in corresponding to given state-action pairs.
     */
    virtual Eigen::VectorXd predict_implementation(
        Eigen::Ref<const Eigen::VectorXd> state,
        const Eigen::Ref<const Eigen::VectorXi>& actions) {
        Eigen::VectorXd td_error;
        td_error = Eigen::VectorXd::Zero(actions.size());

        for (int action_idx = 0; action_idx < actions.size(); action_idx++) {
            int action = actions[action_idx];
            omp_set_lock(const_cast<omp_lock_t*>(&action_locks[action]));
            // Call implementation for single action
            td_error[action_idx] = predict_implementation(state,
                actions[action_idx]);
            omp_unset_lock(const_cast<omp_lock_t*>(&action_locks[action]));
        }

        return td_error;
    }

    /**
     * @brief Updates the value for a given state-action pair.
     * 
     * @param state State vector
     * @param action Action value
     * @param target Target value
     * @return Value error 
     */
    virtual double update_implementation(
        const Eigen::Ref<const Eigen::VectorXd>& state,
        int action,
        double target) {
        throw std::logic_error("Not implemented");
    }

  public:
    const int number_of_actions;        //<! Number of discrete actions
    const int dimensions_of_statespace; //<! Size of the state vector

    /**
     * @brief Construct a new Approximator object
     * 
     * @param number_of_actions Number of discrete actions
     * @param dimensions_of_statespace Size of the state vector
     */
    Approximator(int number_of_actions, int dimensions_of_statespace)
      : number_of_actions(number_of_actions),
        dimensions_of_statespace(dimensions_of_statespace) {
      action_locks.resize(number_of_actions);
      for (auto& lck : action_locks) omp_init_lock(&lck);
    }
    
    /**
     * @brief Destroy the Approximator object
     * 
     */
    virtual ~Approximator() {
        for (auto& lck : action_locks) omp_destroy_lock(&lck);
    }

    /**
     * @brief Saves parameters of estimator to file.
     * 
     * @param filename Name of file to write.
     */
    virtual void save(std::string filename) = 0;

    /**
     * @brief Load parameters of estimator from file.
     * 
     * @param filename Name of file to load.
     */
    virtual void load(std::string filename) = 0;

    /**
     * @brief Predicts the values of multiple state-action pairs.
     * 
     * @param state State vector
     * @param actions Multiple actions to evaluate stored in a vector
     * @return Values of corresponding state-action pairs
     */
    virtual Eigen::VectorXd predict(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const Eigen::Ref<const Eigen::VectorXi>& actions) {
        // Check input arguments
        if (state.size() != dimensions_of_statespace)
            throw std::invalid_argument("State vector has wrong size.");
        if (actions.minCoeff() < 0 || actions.maxCoeff() >= number_of_actions)
            throw std::invalid_argument(
                "Action vector contains illegal values.");

        return predict_implementation(state, actions);
    }

    /**
     * @brief Updates the value for a given state-action pair.
     * 
     * @param state State vector
     * @param action Action value
     * @param target Target state-action value
     * @return Value error 
     */
    virtual double update(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const int action,
      double target) {
        // Check input arguments
        if (state.size() != dimensions_of_statespace)
            throw std::invalid_argument("State vector has wrong size.");
        if (action < 0 || action >= number_of_actions)
            throw std::invalid_argument("Action value is illegal.");

        double td_error;
        omp_set_lock(&action_locks[action]);
        // Call implementation
        td_error = update_implementation(state, action, target);
        omp_unset_lock(&action_locks[action]);
        return td_error;
    }
};

#endif