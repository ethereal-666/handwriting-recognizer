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


add_executable(test_mnist_loader
    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    tests/test_mnist_loader.cpp
)

target_include_directories(test_mnist_loader PRIVATE
    cpp/include
)

add_executable(test_linear
    cpp/src/matrix.cpp
    cpp/src/linear.cpp
    tests/test_linear.cpp
)

target_include_directories(test_linear PRIVATE
    cpp/include
)

add_executable(train
    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    cpp/src/linear.cpp
    cpp/src/softmax.cpp
    cpp/src/cross_entropy.cpp
    cpp/src/train.cpp
)

target_include_directories(train PRIVATE
    cpp/include
)


find_package(Threads REQUIRED)

add_executable(lr_sweep
    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    cpp/src/linear.cpp
    cpp/src/softmax.cpp
    cpp/src/cross_entropy.cpp
    tests/lr_sweep.cpp
)

target_include_directories(lr_sweep PRIVATE
    cpp/include
)

target_link_libraries(lr_sweep PRIVATE
    Threads::Threads
)

add_executable(test_load
    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    cpp/src/linear.cpp
    cpp/src/softmax.cpp
    tests/test_load.cpp
)

target_include_directories(test_load PRIVATE
    cpp/include
)
