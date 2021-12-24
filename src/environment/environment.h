#ifndef __ENVIRONMENT_HPP
#define __ENVIRONMENT_HPP

#include "Eigen/Dense"
#include <string>

/**
 * @brief Base class for learning environments. Provides an interface
 *        for environments with continous state-space and discrete action space.
 * 
 */
class Environment {
  public:
    Environment() {} 

    /**
     * @brief Get the Number Of possible discrete Actions 
     * 
     * @return int 
     */
    virtual int getNumberOfActions() = 0;

    /**
     * @brief Get the number of state-space dimensions
     * 
     * @return int 
     */
    virtual int getStateDim() = 0;

    /**
     * @brief Get the State vector
     * 
     * @return Eigen::VectorXd 
     */
    virtual Eigen::VectorXd getState() = 0;

    /**
     * @brief Perform one step in the environment
     * 
     * @param action Action to take
     * @param observation Output of new observation vector
     * @param done Output of flag that indicates if final state is reached
     */
    virtual void step(int action, Eigen::Ref<Eigen::VectorXd> observation, bool& done) = 0;

    /**
     * @brief Resets the environment
     * 
     * @param observation The new state vector
     */
    virtual void reset(Eigen::Ref<Eigen::VectorXd> observation) = 0;

    /**
     * @brief Resets the environment
     * 
     */
    void reset() { Eigen::VectorXd obs = Eigen::VectorXd::Zero(getStateDim()); reset(obs); }

    /**
     * @brief Visualizes the environment
     * 
     * @param mode Selection for the output device, e.g. "console"
     */
    virtual void render(std::string mode="console") = 0;
};

#endif