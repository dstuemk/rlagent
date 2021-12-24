#ifndef __EPSILON_GREEDY_H_
#define __EPSILON_GREEDY_H_

#include "src/policy/policy.h"
#include <random>

/**
 * @brief Implements an epsilon greedy policy.
 * 
 */
class EpsilonGreedy : public Policy {
  private:
    std::mt19937 random_generator;                      //<! random number generator
    std::uniform_real_distribution<> distribution_real; //<! distribution for greediness
    std::uniform_int_distribution<> distribution_int;   //<! distribution for random action
    Eigen::VectorXi actions;                            //<! Holds all possible action values
  
  public:
    double epsilon; //<! Percentage of randomly taken actions

    /**
     * @brief Construct a new Epsilon Greedy policy
     * 
     * @param epsilon Percentage of randomly taken actions
     * @param approximator Approximater which provides state-action values
     */
    EpsilonGreedy(
        double epsilon,
        const std::shared_ptr<Approximator>& approximator);
    
    int apply(
        const Eigen::Ref<const Eigen::VectorXd>& state) override;
};

#endif