#include "src/environment/flappy_simulator.h"

void FlappySimulator::randomizeSeed() {
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    srand((unsigned int) us);
}

bool FlappySimulator::checkOverlap(double R, double Xc, double Yc,
    double X1, double Y1, double X2, double Y2) {    
    // Find the nearest point on the
    // rectangle to the center of
    // the circle
    double Xn = std::max(X1, std::min(Xc, X2));
    double Yn = std::max(Y1, std::min(Yc, Y2));
        
    // Find the distance between the
    // nearest point and the center
    // of the circle
    // Distance between 2 points,
    // (x1, y1) & (x2, y2) in
    // 2D Euclidean space is
    // ((x1-x2)**2 + (y1-y2)**2)**0.5
    double Dx = Xn - Xc;
    double Dy = Yn - Yc;
    return (Dx * Dx + Dy * Dy) <= R * R;
}

FlappySimulator::FlappySimulator(bool with_gui) 
     : Environment() {
    state = Eigen::VectorXd::Zero(SIZE_OF_STATESPACE);
    Environment::reset();
    
    
    if (!with_gui) {
        window = nullptr;
        return;
    }

    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow(
        "Flappy Simulator",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        screen_width*screen_scale,
        screen_height*screen_scale,
        0
    );

    if (!window) {
        std::cout << "Failed to create window" << std::endl
                  << "SDL Error: " << SDL_GetError() << std::endl;
    }

    window_surface = SDL_GetWindowSurface(window);
    
    if (!window_surface) {
        std::cout << "Failed to get window's surface" << std::endl
                  << "SDL Error: " << SDL_GetError() << std::endl;
    }
}

FlappySimulator::~FlappySimulator() {
    if (window) {
        SDL_FreeSurface(window_surface);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
}

void FlappySimulator::render(std::string mode) {
    if (!window) {
        std::cout << "Can only render in GUI mode" << std::endl;
        return;
    } 
    // Clean
    SDL_FillRect(window_surface, NULL, SDL_MapRGB(window_surface->format, 0, 0, 0));

    // Game scene
    SDL_Rect flappy_position;
    flappy_position.x = (flappy_x - flappy_radius) * screen_scale;
    flappy_position.y = (state[FLAPPY_Y] - flappy_radius) * screen_scale;
    flappy_position.w = flappy_radius * 2.0 * screen_scale;
    flappy_position.h = flappy_radius * 2.0 * screen_scale;
    SDL_FillRect(window_surface, &flappy_position, SDL_MapRGB(window_surface->format, 200, 0, 0));

    double pipe_x = state[PIPE_1_X];
    for (int i=0; i < 2; i++) {
        double pipe_left = pipe_x - pipe_width;
        double pipe_lower_top = state[PIPE_1_Y + i] + pipe_opening / 2.0;
        double pipe_lower_bottom = screen_height;
        double pipe_upper_top = 0.0;
        double pipe_upper_bottom = pipe_lower_top - pipe_opening;

        SDL_Rect pipe_lower_position;
        pipe_lower_position.x = pipe_left * screen_scale;
        pipe_lower_position.y = pipe_lower_top * screen_scale;
        pipe_lower_position.w = pipe_width * screen_scale;
        pipe_lower_position.h = (pipe_lower_bottom - pipe_lower_top) * screen_scale;
        SDL_FillRect(window_surface, &pipe_lower_position, SDL_MapRGB(window_surface->format, 0, 0, 200));

        SDL_Rect pipe_upper_position;
        pipe_upper_position.x = pipe_left * screen_scale;
        pipe_upper_position.y = pipe_upper_top * screen_scale;
        pipe_upper_position.w = pipe_width * screen_scale;
        pipe_upper_position.h = (pipe_upper_bottom - pipe_upper_top) * screen_scale;
        SDL_FillRect(window_surface, &pipe_upper_position, SDL_MapRGB(window_surface->format, 0, 0, 200));

        pipe_x += pipe_distance;
        if (pipe_x > screen_width + pipe_width) pipe_x -= (pipe_distance*2);
    }

    // Draw
    SDL_UpdateWindowSurface(window);
}

void FlappySimulator::play(std::shared_ptr<Policy> policy, double play_time_sec, double speedup) {
    if (!window) {
        std::cout << "Can only play in GUI mode" << std::endl;
        return;
    }

    Environment::reset();

    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    auto frame_ms = start_ms - dt * 1000.0 / speedup;

    // Stores selected action (can be overwriten by keyboard)
    int selected_action = -1;

    bool keep_running = true;
    while(keep_running) {
        // SDL Event handling
        SDL_Event e;
        while (SDL_PollEvent(&e) > 0) {
            switch(e.type) {
                case SDL_QUIT:
                    keep_running = false;
                    break;
            }
        }

        // Holding shift overwrites action (Space is flap action)
        unsigned char const *keys = SDL_GetKeyboardState(nullptr);
        if (keys[SDL_SCANCODE_LSHIFT]) {
            selected_action = keys[SDL_SCANCODE_SPACE] ? 1:0;
            std::cout << "Overwrite action: " << selected_action << std::endl;
        }

        // Get current time
        now = std::chrono::high_resolution_clock::now().time_since_epoch();
        auto current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

        // Check if next frame to draw
        if (current_ms - frame_ms >= dt * 1000.0 / speedup) {
            Eigen::VectorXd obs = getState();
            bool episode_over = false;
            // Choose policy action if no keyboard input
            if (selected_action < 0) selected_action = policy->apply(getState());
            // Perform step and render
            step(selected_action, obs, episode_over);
            std::cout << obs << std::endl;
            std::cout << "Finished: " << episode_over << std::endl;
            render();
            // Check if episode over
            if (episode_over) reset(obs);
            // Reset frame timer and action
            frame_ms = current_ms;
            selected_action = -1;
        }

        // Check if timeout
        keep_running &= (current_ms - start_ms < play_time_sec*1000.0);
    }
}

void FlappySimulator::step(int action, Eigen::Ref<Eigen::VectorXd> observation, bool& done) {
    // We assume constant acceleration in one timestep
    double acceleration = gravity - double(action)*flappy_accel;
    double flappy_v = state[FLAPPY_V] + acceleration*dt;
    // Limit velocity to maximum value
    if (std::abs(flappy_v) > flappy_vmax) {
        flappy_v = flappy_v / std::abs(flappy_v) * flappy_vmax;
    }
    double flappy_y = state[FLAPPY_Y] + state[FLAPPY_V]*dt + 0.5*acceleration*std::pow(dt,2.0);

    // Move pipes to the left
    double pipe_1_x = state[PIPE_1_X] - flappy_speed*dt;
    double pipe_1_y = state[PIPE_1_Y];
    double pipe_2_y = state[PIPE_2_Y];
    if (pipe_1_x < 0.0){
        pipe_1_x += pipe_distance * 2.0;
        pipe_1_y = double(rand())/double(RAND_MAX) * (screen_height - pipe_opening*1.5) + pipe_opening*0.75;
    } 
    if (pipe_1_x - pipe_distance < 0.0 && pipe_1_x - pipe_distance >= -flappy_speed*dt) {
        pipe_2_y = double(rand())/double(RAND_MAX) * (screen_height - pipe_opening*1.5) + pipe_opening*0.75;
    }

    // Check for pipe collisions
    collision = false;
    double pipe_x = pipe_1_x;
    for (int i=0; i < 2; i++) {
        double pipe_right = pipe_x;
        double pipe_left = pipe_x - pipe_width;
        double pipe_lower_top = state[PIPE_1_Y + i] + pipe_opening / 2.0;
        double pipe_lower_bottom = screen_height;
        double pipe_upper_top = 0.0;
        double pipe_upper_bottom = pipe_lower_top - pipe_opening;

        collision = collision || checkOverlap(
            flappy_radius, flappy_x, flappy_y, 
            pipe_left, pipe_lower_top, 
            pipe_right, pipe_lower_bottom);
            
        collision = collision || checkOverlap(
            flappy_radius, flappy_x, flappy_y, 
            pipe_left, pipe_upper_top, 
            pipe_right, pipe_upper_bottom);
                        
        pipe_x += pipe_distance;
        if (pipe_x > screen_width + pipe_width) pipe_x -= (pipe_distance*2);
    }

    // Check for ground / ceiling collision
    collision = collision || (flappy_y + flappy_radius >= screen_height);
    collision = collision || (flappy_y - flappy_radius <= 0.0);
        
    // If we had collsion reset movement
    if (collision) {
        flappy_y = state[FLAPPY_Y];
        flappy_v = 0.0;
    }

    // Set done flag
    done = collision;

    // Update state
    state[PIPE_1_X] = pipe_1_x;
    state[PIPE_1_Y] = pipe_1_y;
    state[PIPE_2_Y] = pipe_2_y;
    state[FLAPPY_V] = flappy_v;
    state[FLAPPY_Y] = flappy_y;

    // Output variable
    observation = state;
}

void FlappySimulator::reset(Eigen::Ref<Eigen::VectorXd> observation) {
    randomizeSeed();
    state = (Eigen::MatrixXd::Random(SIZE_OF_STATESPACE, 1).array() + 1.0) / 2.0;
    //state[PIPE_1_X] = state[PIPE_1_X] * screen_width;
    state[PIPE_1_X] = flappy_x - flappy_radius + double(rand() % 2) * pipe_distance;
    state[PIPE_1_Y] = state[PIPE_1_Y] * (screen_height - pipe_opening * 1.5) + pipe_opening * 0.75;
    state[PIPE_2_Y] = state[PIPE_2_Y] * (screen_height - pipe_opening * 1.5) + pipe_opening * 0.75;
    //state[FLAPPY_Y] = state[FLAPPY_Y] * (screen_height - 4.0 * flappy_radius) + 2.0 * flappy_radius;
    state[FLAPPY_Y] = screen_height / 2.0;
    state[FLAPPY_V] = 0.0;
    collision = false;
    observation = state;
}