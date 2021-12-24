cmake_minimum_required(VERSION 3.5.1)
project(rlagent)

set(SOURCE
        # All source files here
        ${CMAKE_CURRENT_SOURCE_DIR}/environment/flappy_simulator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/approximator/state_aggregation.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/policy/epsilon_greedy.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/learner/sarsa.cc
        
        )

set(HEADER
        # All header files here
        ${CMAKE_CURRENT_SOURCE_DIR}/environment/environment.h
        ${CMAKE_CURRENT_SOURCE_DIR}/environment/flappy_simulator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/approximator/approximator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/approximator/state_aggregation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/policy/policy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/policy/epsilon_greedy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/learner/learner.h
        ${CMAKE_CURRENT_SOURCE_DIR}/learner/sarsa.h
        )

set(HEADER ${HEADER} PARENT_SCOPE)

set(SOURCE ${SOURCE} PARENT_SCOPE)