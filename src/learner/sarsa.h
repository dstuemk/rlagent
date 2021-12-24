#ifndef __SARSA_H_
#define __SARSA_H_

#include <memory>
#include "src/learner/learner.h"
#include "src/policy/policy.h"

/**
 * @brief Implementation of a n-step SARSA algorithm
 * 
 */
class Sarsa : public Learner {
  public:
    const double discount;                //<! Discount factor
    const std::shared_ptr<Policy> policy; //<! Policy to learn
    const int n_steps;                    //<! N-Steps parameter of n-step SARSA

   /**
   * Constructor for the Sarsa algorithm class
   * @brief Constructor for the SARSA algorithm
   * @param discount The discount factor gamma
   * @param policy The policy that is used in the algorithm
   * @param approximator The approximator that estimates the value function
   * @param reward The reward function to be used
   * @param environment_generator Function which generates environment  to use
   * @param n_steps Number of steps for n-step SARSA
   */
    explicit Sarsa(
        double const& discount,
        std::shared_ptr<Policy> const& policy,
        std::shared_ptr<Approximator> const& approximator,
        reward_function const& reward,
        environment_function const& environment_generator,
        int n_steps = 1);

    
    std::shared_ptr<Policy> get_policy() override;

  protected:
    void learn_episode(
        int max_steps,
        double* ssve_out,
        double* total_reward_out,
        Environment* environment) override;
};

#endif