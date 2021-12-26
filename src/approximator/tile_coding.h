#ifndef __TILE_CODING_H_
#define __TILE_CODING_H_

#include "src/approximator/approximator.h"
#include "src/approximator/state_aggregation.h"

class TileCoding : public Approximator {
  public:
    double step_size; //<! How much the updates affect the values
    
    /**
     * @brief Construct a new Tile coding object
     * 
     * @param number_of_actions Number of discrete actions
     * @param dimensions_of_statespace Size of state-space vector
     * @param step_size Step size, also called learning rate
     * @param tilings Number of tiling layers
     * @param displacement Displacement vector for each layer
     * @param segments Number of segments for each state dimension
     * @param min_values Minimum values of state-space
     * @param max_values Maximum values of state-space
     * @param action_kernel Defines the influence of an action to its neighboring actions
     * @param init_min_value Minimum value for random initialization
     * @param init_max_value Maximum value for random initialization
     */
    TileCoding(
        int number_of_actions,
        int dimensions_of_statespace,
        double step_size,
        int tilings,
        const Eigen::Ref<const Eigen::VectorXi> &displacement,
        const Eigen::Ref<const Eigen::VectorXi> &segments,
        const Eigen::Ref<const Eigen::VectorXf> &min_values,
        const Eigen::Ref<const Eigen::VectorXf> &max_values,
        const Eigen::Ref<const Eigen::VectorXf> &action_kernel = (Eigen::Matrix<float, 1, 1>()<< 1.0).finished(),
        double init_min_value = 0.0, double init_max_value = 0.0);
    
    void save(std::string filename);

    void load(std::string filename);

    Eigen::VectorXd predict(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const Eigen::Ref<const Eigen::VectorXi>& actions) override;
    
    double update(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const int action,
      double target) override;
  
  private:
    std::vector<StateAggregation> layers;

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