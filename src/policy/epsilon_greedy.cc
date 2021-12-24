#include "src/policy/epsilon_greedy.h"
#include <utility>

EpsilonGreedy::EpsilonGreedy(
    double epsilon,
    const std::shared_ptr<Approximator>& approximator)
    : Policy(approximator),
    random_generator(std::random_device()()),
    distribution_real(0, 1),
    distribution_int(0, approximator->number_of_actions-1),
    epsilon(epsilon) {
        actions = Eigen::VectorXi::LinSpaced(
            approximator->number_of_actions,
            0, approximator->number_of_actions-1);
}

int EpsilonGreedy::apply(
    const Eigen::Ref<const Eigen::VectorXd>& state) {
    if (distribution_real(random_generator) < 1.0 - epsilon) {
        auto q_values = approximator->predict(state, actions);
        int index = 0;
        for (int i = 1; i < q_values.size(); i++) {
            if (q_values[i] > q_values[index]) index = i;
        }
        // q_values.maxCoeff(&index);
        return index;
    }
    return distribution_int(random_generator);
}