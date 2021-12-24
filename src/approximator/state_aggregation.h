#ifndef __STATE_AGGREGATION_H_
#define __STATE_AGGREGATION_H_

#include "src/approximator/approximator.h"

class StateAggregation : public Approximator {
  public:
    double step_size; //<! How much the updates affect the values

    /**
     * @brief Construct a new State Aggregation object
     * 
     * @param number_of_actions Number of discrete actions
     * @param dimensions_of_statespace Size of state-space vector
     * @param step_size Step size, also called learning rate
     * @param segments Number of segments for each state dimension
     * @param min_values Minimum values of state-space
     * @param max_values Maximum values of state-space
     * @param action_kernel Defines the influence of an action to its neighboring actions
     * @param init_min_value Minimum value for random initialization
     * @param init_max_value Maximum value for random initialization
     */
    StateAggregation(
        int number_of_actions,
        int dimensions_of_statespace,
        double step_size,
        const Eigen::Ref<const Eigen::VectorXi> &segments,
        const Eigen::Ref<const Eigen::VectorXf> &min_values,
        const Eigen::Ref<const Eigen::VectorXf> &max_values,
        const Eigen::Ref<const Eigen::VectorXf> &action_kernel = (Eigen::Matrix<float, 1, 1>()<< 1.0).finished(),
        double init_min_value = 0.0, double init_max_value = 0.0);

    void save(std::string filename);

    void load(std::string filename);

  private:
    Eigen::VectorXf action_kernel;
    Eigen::VectorXf values;       //<! Storage for state-action values
    Eigen::VectorXi segments;     //<! Number of segments for each state dimension
    Eigen::VectorXf segment_size; //<! Size of each segment in state-space
    Eigen::VectorXf min_values;   //<! Minimum state-space values
    Eigen::VectorXf max_values;   //<! Maximum state-space values

    /**
     * @brief Get the indices of the state-action values for all possible actions.
     * 
     * @param state State vector
     * @return Indices for state-action values
     */
    Eigen::VectorXi get_indices(Eigen::VectorXd state);

  protected:
    double predict_implementation(
        Eigen::Ref<const Eigen::VectorXd> state,
        int action) override;

    double update_implementation(
        const Eigen::Ref<const Eigen::VectorXd>& state,
        int action,
        double target) override;
};



#endif