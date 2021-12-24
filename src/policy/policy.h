#ifndef __POLICY_H_
#define __POLICY_H_

#include "src/approximator/approximator.h"
#include <memory>
#include <utility>

/**
 * @brief Base class for policies. Restricted to discrete
 *        action space.
 */
class Policy {
  protected:
    std::shared_ptr<Approximator> approximator; //<! Approximater which provides state-action values

  public:
    /**
     * @brief Construct a new Policy
     * 
     * @param approximator Approximater which provides state-action values
     */
    Policy(
        std::shared_ptr<Approximator> approximator)
        : approximator(std::move(approximator)) {}

    /**
     * @brief Return an action value based on the given state vector
     * 
     * @param state State vector
     * @return int 
     */
    virtual int apply(
        const Eigen::Ref<const Eigen::VectorXd>& state) = 0;
};

#endif