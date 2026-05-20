cmake_minimum_required(VERSION 3.16)

project(face_door_demo_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(face_door_demo_cpp
    src/main.cpp
)

target_compile_options(face_door_demo_cpp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
