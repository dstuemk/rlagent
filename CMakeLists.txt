cmake_minimum_required(VERSION 3.5.1)
project(rlagent)

# General C++ Flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Werror") # Add -pg here for profiling
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenMP REQUIRED)

# Include source code
add_subdirectory(src)
include_directories(src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Include SDL
set( SDL_STATIC ON CACHE BOOL "" FORCE )
set( SDL_SHARED OFF CACHE BOOL "" FORCE )
add_subdirectory(sdl)
include_directories(${SDL2_INCLUDE_DIRS})

# Include Eigen
include_directories(eigen)

add_executable(
    ${PROJECT_NAME}
    ${SOURCE} 
    ${HEADER} 
    main.cc
    utils.cc)
target_link_libraries(
    ${PROJECT_NAME}
    SDL2-static
    OpenMP::OpenMP_CXX)
