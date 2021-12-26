#include "src/approximator/tile_coding.h"

TileCoding(
    int number_of_actions,
    int dimensions_of_statespace,
    double step_size,
    int tilings,
    const Eigen::Ref<const Eigen::VectorXi> &displacement,
    const Eigen::Ref<const Eigen::VectorXi> &segments,
    const Eigen::Ref<const Eigen::VectorXf> &min_values,
    const Eigen::Ref<const Eigen::VectorXf> &max_values,
    const Eigen::Ref<const Eigen::VectorXf> &action_kernel,
    double init_min_value, double init_max_value)
  : Approximator(number_of_actions, dimensions_of_statespace) {
      // How big is each dimension
      auto size_statespace = max_values - min_values;

      // How big is each segment
      auto segment_size = size_statespace.array() / segments.cast<float>().array();

      // How big is a "fundamental" tile
      auto tile_size = segment_size.array() / float(tilings);

      for (int i=0; i < tilings; i++) {

          auto offset = tile_size.array() * displacement.cast<float>().array() * float(i);
          auto layer_min_values = min_values.array() - offset.array();
          auto layer_segments = segments.cast<float>().array() + (offset.array() / segment_size.array()).ceil();
          auto layer_max_values = layer_min_values.array() + segment_size.array() * layer_segments.array();

          layers.push_back(StateAggregation(
              number_of_actions,
              dimensions_of_statespace,
              step_size / tilings,
              layer_segments,
              layer_min_values,
              layer_max_values,
              action_kernel,
              init_min_value / tilings, 
              init_max_value / tilings
          ));
      }
}

Eigen::VectorXd TileCoding::predict(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const Eigen::Ref<const Eigen::VectorXi>& actions) {
    Eigen::VectorXd prediction = Eigen::VectorXd::Zero(dimensions_of_statespace);
    for(auto& layer: layers) {
        prediction = prediction + layer.predict(state,actions) / layers.size();
    }
    return prediction;
}
    
double TileCoding::update(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const int action,
      double target) {
    double td_error = 0.0;
    for(auto& layer: layers) {
        td_error = td_error + layer.update(state, action, target) / layers.size();
    }
    return td_error;
}

double TileCoding::predict_implementation(
        Eigen::Ref<const Eigen::VectorXd> state,
        int action) {
    throw std::logic_error("Not implemented");
}

double TileCoding::update_implementation(
        const Eigen::Ref<const Eigen::VectorXd>& state,
        int action,
        double target) {
    throw std::logic_error("Not implemented");
}