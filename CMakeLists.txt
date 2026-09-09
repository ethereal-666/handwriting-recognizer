cmake_minimum_required(VERSION 3.16)

project(handwriting_recognizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(test_matrix
    cpp/src/matrix.cpp
    tests/test_matrix.cpp
)

target_include_directories(test_matrix PRIVATE
    cpp/include
)