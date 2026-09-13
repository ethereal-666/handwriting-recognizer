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

add_executable(train_linear
    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    cpp/src/linear.cpp
    cpp/src/softmax.cpp
    cpp/src/cross_entropy.cpp
    cpp/src/train_linear.cpp
)

target_include_directories(train_linear PRIVATE
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

add_executable(test_recognizer
    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    cpp/src/linear.cpp
    cpp/src/softmax.cpp
    cpp/src/recognizer.cpp
    cpp/src/relu.cpp
    tests/test_recognizer.cpp
)

target_include_directories(test_recognizer PRIVATE
    cpp/include
)

add_library(recognizer SHARED
    cpp/src/matrix.cpp
    cpp/src/linear.cpp
    cpp/src/softmax.cpp
    cpp/src/recognizer.cpp
    cpp/src/recognizer_api.cpp
    cpp/src/relu.cpp
)

target_include_directories(recognizer PRIVATE
    cpp/include
)

add_executable(train_mlp
    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    cpp/src/linear.cpp
    cpp/src/softmax.cpp
    cpp/src/cross_entropy.cpp
    cpp/src/relu.cpp
    cpp/src/train_mlp.cpp
)

target_include_directories(train_mlp PRIVATE
    cpp/include
)

add_executable(
    lr_search_mlp

    tests/lr_search_mlp.cpp

    cpp/src/matrix.cpp
    cpp/src/mnist_loader.cpp
    cpp/src/linear.cpp
    cpp/src/relu.cpp
    cpp/src/softmax.cpp
    cpp/src/cross_entropy.cpp
)

target_include_directories(lr_search_mlp PRIVATE
    cpp/include
)

target_link_libraries(lr_search_mlp PRIVATE
    Threads::Threads
)