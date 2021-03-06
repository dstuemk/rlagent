#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "src/environment/flappy_simulator.h"
#include "src/learner/sarsa.h"
#include "src/policy/epsilon_greedy.h"
#include "src/approximator/tile_coding.h"

#include "utils.h"

void learn_batch(Learner* learner, int batch_size, int episode_length,
  std::vector<double>& msve_batch, std::vector<double>& reward_batch);

void save_statistics(std::string filename, const double* msve_array,
  const double* reward_array, int number_of_episodes);

int main(int argc, char** argv) {
  // Parse command line arguments
  const char* execution_mode = get_cmd_option(
    argv, argv+argc, "-exec");
  bool mode_learn = false;
  bool mode_play = false;
  if (execution_mode) {
    mode_learn = std::string(execution_mode) == "learn";
    mode_play = std::string(execution_mode) == "play";
  }

  const char* working_directory = get_cmd_option(
    argv, argv+argc, "-wdir");
  if (!working_directory) {
    working_directory = "./";
  }

  // Initialize environment
  FlappySimulator env(true);

  // Create value function approximator
  const double learning_rate = 1e-1;
  const int tilings = 5;
  auto displacement = (Eigen::Matrix<int, 5, 1>()
    << 1, 3, 5, 7, 11).finished();
  auto state_space_segments = (Eigen::Matrix<int, 5, 1>()
    << 10, 10, 10, 10, 10).finished();
  auto state_space_min = (Eigen::Matrix<float, 5, 1>()
    << 0, 3.75, 3.75, 1, -10).finished();
  auto state_space_max = (Eigen::Matrix<float, 5, 1>()
    << 11, 10.25, 10.25, 13, 10).finished();
  auto approximator = std::make_shared<TileCoding>(
                env.getNumberOfActions(),
                env.getStateDim(),
                learning_rate,
                tilings,
                displacement,
                state_space_segments,
                state_space_min,
                state_space_max);
  
  // Create policy
  double epsilon = 0.2;
  double epsilon_decay = 1.0 - 3e-6;
  auto policy = std::make_shared<EpsilonGreedy>(epsilon, approximator);

  // Create learner
  const int number_of_episodes = 1e6;
  Learner::reward_function reward = [](Eigen::VectorXd x, int a, Eigen::VectorXd x_next, Environment* env) {
      auto* flappy_env = (FlappySimulator*)env;
      return flappy_env->getCollision() ? -100.0 : 1.0;
  };
  Learner::environment_function init_env = []() {
    auto env = std::make_shared<FlappySimulator>();
    return env;
  };
  const double discount = 0.9;
  Sarsa learner(discount, policy, approximator, reward, init_env, 20);

  if (mode_play) {
    // Load pretrained approximator
    approximator->load(std::string(working_directory) + "/approximator.dat");
    
    // Perform epsilon decay process
    policy->epsilon = policy->epsilon * std::pow(epsilon_decay, number_of_episodes);

    // Play for 5 minutes
    env.play(policy, 300.0);
  }

  if (mode_learn) {
    // Learning phase
    const int episode_length = 400;
    learner.verbose = true;
    
    // This could take a while ...
    const int number_of_batches = 100;
    int current_batch = 0;
    int remaining_episodes = number_of_episodes;
    while (remaining_episodes > 0) {
      std::cout << "batch number: " << ++current_batch << std::endl;
      // We learn a batch of episodes with fixed parameters
      int batch_size = std::min(
        std::ceil(double(number_of_episodes) / number_of_batches), 
        double(remaining_episodes));
      std::vector<double> msve_batch, reward_batch;
      learn_batch(&learner, batch_size, episode_length, msve_batch, reward_batch);      
      // Save parameters
      approximator->save(std::string(working_directory) + "/approximator.dat");
      // Save statistics
      save_statistics(std::string(working_directory) + "/statistics.csv",
        msve_batch.data(), reward_batch.data(), batch_size);
      // Play one example game
      env.play(policy, 10.0);
      // Decay process of epsilon
      policy->epsilon = policy->epsilon * std::pow(epsilon_decay, batch_size);
      // Next batch ...
      remaining_episodes -= batch_size;  
    }    
  }
  
  // Fin.
  std::cout << "finished" << std::endl;
  return 0;
}


void learn_batch(Learner* learner, int batch_size, int episode_length,
  std::vector<double>& msve_batch, std::vector<double>& reward_batch) {
    // Output batch info
    std::cout << "batch size: " << batch_size << std::endl
              << std::endl;
    // Learn
    learner->learn(batch_size, episode_length, msve_batch, reward_batch);
    // Output batch stats
    std::cout << "batch mean reward: " 
              << std::accumulate(reward_batch.begin(), reward_batch.end(), 0.0) / reward_batch.size() << std::endl
              << "batch msve: "
              << std::accumulate(msve_batch.begin(), msve_batch.end(), 0.0) / msve_batch.size() << std::endl;
}


void save_statistics(std::string filename, const double* msve_array,
  const double* reward_array, int number_of_episodes) {
    std::fstream stats;
    stats.open(filename, std::ios_base::app | std::ios_base::ate);
    // Output in CSV format
    if (!stats.tellp()) {
      std::cout << "Create file: " << filename << std::endl;
      stats << "MSVE,REWARD\n";
    }
    std::cout << "Write into file: " << filename << std::endl;
    for (int i=0; i < number_of_episodes; i++) {
      stats << msve_array[i] << "," << reward_array[i] << "\n";
    }
    stats.close();
}