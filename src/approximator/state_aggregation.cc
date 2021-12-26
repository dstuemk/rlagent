#include "src/approximator/state_aggregation.h"
#include <fstream>

StateAggregation::StateAggregation(
        int number_of_actions,
        int dimensions_of_statespace,
        double step_size,
        const Eigen::Ref<const Eigen::VectorXi> &segments,
        const Eigen::Ref<const Eigen::VectorXf> &min_values,
        const Eigen::Ref<const Eigen::VectorXf> &max_values,
        const Eigen::Ref<const Eigen::VectorXf> &action_kernel,
        double init_min_value, double init_max_value)
        : Approximator(
            number_of_actions,
            dimensions_of_statespace),
        step_size(step_size),
        action_kernel(action_kernel),
        segments(segments),
        min_values(min_values),
        max_values(max_values) {
    // How big is each dimension
    auto size_statespace = max_values - min_values;

    // How big is each segment
    segment_size = size_statespace.array() / segments.cast<float>().array();
    
    // Reserve enough memory
    values = (Eigen::VectorXf::Random(
        segments.prod() * number_of_actions).array() + 1.0) / 2.0
        * (init_max_value - init_min_value) + init_min_value;
}

Eigen::Ref<Eigen::VectorXf> StateAggregation::getValues() {
    return values;
}

void StateAggregation::save(std::string filename) {
    std::ofstream outfile(filename, std::ios_base::binary);
    if (outfile.is_open()) {
        outfile.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<int64_t>(
                values.size() * sizeof(values.data()[0])));
        outfile.close();
    }
}

void StateAggregation::load(std::string filename) {
    std::ifstream infile(filename, std::ios_base::binary);
    if (infile.good()) {
        infile.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<int64_t>(
                values.size() * sizeof(values.data()[0])));
        infile.close();
    }
}

double StateAggregation::predict_implementation(
        Eigen::Ref<const Eigen::VectorXd> state,
        int action) {
    Eigen::VectorXi indices = get_indices(state);
    double prediction = action_kernel[0] * values[indices[action]];
    for (int i=1; i < action_kernel.size(); i++) {
       int action_p = std::min(action + i, number_of_actions-1);
       int action_n = std::max(action - i, 0);
       prediction += action_kernel[i] * values[indices[action_p]]
           + action_kernel[i] * values[indices[action_n]];
    }
    return prediction;
}

double StateAggregation::update_implementation(
        const Eigen::Ref<const Eigen::VectorXd>& state,
        int action,
        double target) {
    // Indices is a vector of form [idx(state,a=0), idx(state,a=1), ..., idx(state,a=A)]
    Eigen::VectorXi indices = get_indices(state);
    // Predict value (action-kernel defines the influence of "neigboring" actions)
    double prediction = action_kernel[0] * values[indices[action]];
    for (int i=1; i < action_kernel.size(); i++) {
       int action_p = std::min(action + i, number_of_actions-1);
       int action_n = std::max(action - i, 0);
       prediction += action_kernel[i] * values[indices[action_p]]
          + action_kernel[i] * values[indices[action_n]];
    }
    // Calculate error
    double prediction_error = target - prediction;
    // Update values
    values[indices[action]] += action_kernel[0] * prediction_error * step_size;
    for (int i=1; i < action_kernel.size(); i++) {
       int action_p = std::min(action + i, number_of_actions-1);
       int action_n = std::max(action - i, 0);
       values[indices[action_p]] += 
           action_kernel[i] * prediction_error * step_size;
       values[indices[action_n]] += 
           action_kernel[i] * prediction_error * step_size;
    }

    return prediction_error;
}

Eigen::VectorXi StateAggregation::get_indices(Eigen::VectorXd state) {
    Eigen::VectorXi indices_out = Eigen::VectorXi::Zero(number_of_actions);
    Eigen::VectorXf state_shifted = state.cast<float>() - min_values;
    Eigen::VectorXi indices = (state_shifted.array() / segment_size.array()).cast<int>();
    indices = indices.array().min(segments.array() - 1);
    unsigned int index = indices[0];
    for (int i=1; i < indices.size(); i++) {
        index *= segments[i];
        index += indices[i];
    }
    // Also include action
    for (int i=0; i < number_of_actions; i++) {
        indices_out.coeffRef(i) = index + i*segments.prod();
    }
    return indices_out;
}