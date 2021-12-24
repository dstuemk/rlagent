#ifndef __LEARNER_H_
#define __LEARNER_H_

#include <omp.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>
#include <time.h>
#include "src/approximator/approximator.h"
#include "src/policy/policy.h"
#include "src/environment/environment.h"

/**
 * @brief Base class for learning algorithms. Restricted to discrete
 *        action spaces. 
 */
class Learner {
  public:
      typedef std::function<double(Eigen::VectorXd, int, Eigen::VectorXd, Environment*)> reward_function; //!> Reward function, (state, action, state_next, env) -> reward
      typedef std::function<std::shared_ptr<Environment>(void)> environment_function;                     //!> Environment function, provides learning environment instance

      bool verbose;                                     //<! Enable or disable console messages
      const std::shared_ptr<Approximator> approximator; //<! Value function approximator
      reward_function reward;                           //<! Reward function
      environment_function environment_generator;       //<! Environment generator function
  
      /**
       * @brief Construct a new Learner object
       * 
       * @param approximator Value function approximator to use
       * @param reward Reward function to use
       * @param environment_generator Function that generates the learning environment
       */
      Learner(
        std::shared_ptr<Approximator> approximator,
        reward_function reward,
        environment_function environment_generator)
        : verbose(false), approximator(std::move(approximator)),
          reward(std::move(reward)), environment_generator(
            std::move(environment_generator)) {}

      /**
       * @brief Get the (learned) policy
       * 
       * @return std::shared_ptr<Policy> 
       */
      virtual std::shared_ptr<Policy> get_policy() {
          throw std::logic_error("Not implemented");
      }

    /**
     * @brief Learning procedure
     * 
     * @param episodes Number of episodes to learn
     * @param max_steps_per_episode Duration of each episode
     * @param msve_per_episode_out Output of mean square value errors oberserved
     * @param total_reward_per_episode_out Output of total rewards observed
     */
    void learn(
      int episodes,
      int max_steps_per_episode,
      std::vector<double>& msve_per_episode_out,
      std::vector<double>& total_reward_per_episode_out) {
        msve_per_episode_out.resize(episodes);
        total_reward_per_episode_out.resize(episodes);
        time_t last_msg = time(0) - 5;
        #pragma omp parallel for ordered schedule(dynamic, 1)
        for (int episode = 0; episode < episodes; episode++) {
          double ssve_buffer = 0;
          double total_reward_buffer = 0.0;
          auto environment = environment_generator();
          environment->reset();
          learn_episode(
            max_steps_per_episode,
            &ssve_buffer,
            &total_reward_buffer,
            environment.get());
          #pragma omp ordered
          {
            double msve = ssve_buffer / max_steps_per_episode;
            msve_per_episode_out[episode] = msve;
            total_reward_per_episode_out[episode] = total_reward_buffer;
            if (verbose && time(0) - last_msg > 5) {
              last_msg += 5;
              std::cout << ".--------------------------------------." << std::endl
                        << "| Ep: " << episode                        << std::endl 
                        << "| SVE mean: " << msve                     << std::endl 
                        << "| Reward total: " << total_reward_buffer  << std::endl
                        << "'......................................'" << std::endl
                        << std::endl << std::flush;
            }
          }
        }
    }

 protected:
   /**
    * @brief Implements one learning procedure for one episode
    * 
    * @param max_steps Number of steps to run
    * @param ssve_out Output of sum of square value errors
    * @param total_reward_out Output of total reward
    * @param environment Pointer to the environment to learn in
    */
   virtual void learn_episode(
      int max_steps,
      double* ssve_out,
      double* total_reward_out,
      Environment* environment) = 0;
};

#endif 