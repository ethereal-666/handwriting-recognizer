#include "linear.hpp"

#include <cassert>
#include <iostream>

int main() {
  linear layer{3, 2};

  Matrix input{3, 1};

  input.setVal(0, 0, 1.0f);
  input.setVal(1, 0, 2.0f);
  input.setVal(2, 0, 3.0f);

  Matrix output = layer.forward(input);

  assert(output.getRows() == 2);
  assert(output.getCols() == 1);

  std::cout << "output:\n";
  output.printMatrix();

  std::cout << "[PASS] Linear forward\n";

  return 0;
}