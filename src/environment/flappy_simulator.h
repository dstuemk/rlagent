#ifndef __FLAPPY_SIMULATOR_H_
#define __FLAPPY_SIMULATOR_H_

#include "src/environment/environment.h"
#include "src/policy/policy.h"
#include "SDL.h"
#include <chrono>
#include <cstdlib>
#include <iostream>

class FlappySimulator : public Environment {
  private:
    SDL_Window* window;
    SDL_Surface* window_surface;

    Eigen::VectorXd state;
    bool collision;

    // Call this before generating random data
    void randomizeSeed();

    // Source: https://www.geeksforgeeks.org/check-if-any-point-overlaps-the-given-circle-and-rectangle/
    bool checkOverlap(double R, double Xc, double Yc,
      double X1, double Y1, double X2, double Y2);

  public:
    static const int SIZE_OF_STATESPACE = 5;
    static const int PIPE_1_X = 0;
    static const int PIPE_1_Y = 1;
    static const int PIPE_2_Y = 2;
    static const int FLAPPY_Y = 3;
    static const int FLAPPY_V = 4;

    static constexpr double screen_scale  = 30.0;
    static constexpr double screen_width  = 9.0;
    static constexpr double screen_height = 14.0;
    static constexpr double pipe_distance = 6.0;
    static constexpr double pipe_opening  = 5.0;
    static constexpr double pipe_width    = 2.0;
    static constexpr double flappy_radius = 1.0;
    static constexpr double flappy_accel  = 100.0;
    static constexpr double flappy_speed  = 2.0;
    static constexpr double flappy_vmax   = 10.0;
    static constexpr double flappy_x      = 4.0;
    static constexpr double gravity       = 9.81;
    static constexpr double dt            = 1.0 / 20.0;

    FlappySimulator(bool with_gui = false);

    ~FlappySimulator();

    void render(std::string mode="graphic");

    void play(std::shared_ptr<Policy> policy, double play_time_sec = 10.0, double speedup = 1.0);

    int getNumberOfActions() { return 2; }
    
    int getStateDim() { return SIZE_OF_STATESPACE; }

    Eigen::VectorXd getState() { return state; }

    bool getCollision() { return collision; }

    void step(int action, Eigen::Ref<Eigen::VectorXd> observation, bool& done);

    void reset(Eigen::Ref<Eigen::VectorXd> observation);
};

#endif