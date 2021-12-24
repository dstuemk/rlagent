#include "src/learner/sarsa.h"
#include <deque>
#include <utility>
#include <algorithm>
#include <vector>

Sarsa::Sarsa(
        double const& discount,
        std::shared_ptr<Policy> const& policy,
        std::shared_ptr<Approximator> const& approximator,
        reward_function const& reward,
        environment_function const& environment_generator,
        int n_steps) :
    Learner(approximator, reward, environment_generator),
    discount(discount), policy(policy), n_steps(n_steps) {
}

std::shared_ptr<Policy> Sarsa::get_policy() {
    return policy;
}

void Sarsa::learn_episode(
        int max_steps,
        double* ssve,
        double* total_reward,
        Environment* environment) {
    // Storing previous states
    Eigen::MatrixXd n_step_states = Eigen::MatrixXd::Zero(
        environment->getStateDim(), n_steps);
    // Storing previous rewards
    std::vector<double> n_step_rewards(n_steps);
    // Storing previous actions
    std::vector<int> n_step_actions(n_steps);
    // Reset environment and get initial state / action
    Eigen::VectorXd state = Eigen::VectorXd::Zero(environment->getStateDim());
    environment->reset(state);
    int action = policy->apply(state);
    int step = 0;
    int tau = 0;
    *ssve = 0;
    *total_reward = 0;
    // Keep track of remaining steps
    int remaining_steps = max_steps;
    // Store initial state and action
    n_step_states.col(step % n_steps) = state;
    n_step_actions[step % n_steps] = action;
    // Outer loop: Based on remaining steps
    while(remaining_steps > 0) {
        // Inner loop: Based on finite horizont
        do {
            if (step < max_steps) {
                // Perform action in environment
                bool terminal = false;
                Eigen::VectorXd next_state = Eigen::VectorXd::Zero(state.size());
                environment->step(action, next_state, terminal);
                double reward_value = reward(state, action, next_state, environment);
                *total_reward = *total_reward + reward_value;
                // Store next reward and next state
                n_step_rewards[(step+1) % n_steps] = reward_value;
                n_step_states.col((step+1) % n_steps) = next_state;
                // Prepare next iteration
                state = next_state;
                action = policy->apply(state);
                if (terminal) {
                    environment->reset(state);
                    action = policy->apply(state);
                    max_steps = step + 1;
                }
                else {
                    // Store next action
                    n_step_actions[(step+1) % n_steps] = action;
                }
            }

            // Tau is time index we want to update
            tau = step - n_steps + 1;
            if (tau >= 0) {
                double reward_sum = 0.0;
                // accumulate discounted one step rewards
                for (int i=tau+1; i <= std::min(max_steps, step + 1); i++) {
                    double future_reward = n_step_rewards[i % n_steps];
                    double dampening = std::pow(discount, i-tau-1);
                    reward_sum = reward_sum + dampening*future_reward;
                }
                // add expected future reward
                int future_time = tau + n_steps;
                if (future_time < max_steps) {
                    auto future_action = Eigen::Map<Eigen::Matrix<int, 1, 1>>(
                        &n_step_actions[future_time % n_steps]);
                    auto future_state = n_step_states.col(future_time % n_steps);
                    reward_sum = reward_sum 
                        + std::pow(discount, n_steps)*approximator->predict(
                            future_state, future_action)[0];
                }
                // perform update
                double td_error = approximator->update(
                    n_step_states.col(tau % n_steps),
                    n_step_actions[tau % n_steps],
                    reward_sum);
                *ssve = *ssve + std::pow(td_error, 2.0);
            }

            // Increase step
            step = step + 1;

        } while(tau != max_steps - 1);

        // Decrease remaining steps
        remaining_steps -= max_steps;

        // Adapt maximum steps for inner loop also
        max_steps = remaining_steps;
        step = 0;
    }
    
}
