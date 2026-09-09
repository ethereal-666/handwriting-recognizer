#include "matrix.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

bool equalFloat(float a, float b) {
  return std::fabs(a - b) < 1e-6;
}

int main() {

  // =========================
  // 1. 基础构造 / get / set
  // =========================
  Matrix a{2, 3};

  a.setVal(0, 0, 1);
  a.setVal(0, 1, 2);
  a.setVal(0, 2, 3);
  a.setVal(1, 0, 4);
  a.setVal(1, 1, 5);
  a.setVal(1, 2, 6);

  assert(a.getRows() == 2);
  assert(a.getCols() == 3);

  assert(equalFloat(a.getVal(0, 0), 1));
  assert(equalFloat(a.getVal(0, 2), 3));
  assert(equalFloat(a.getVal(1, 0), 4));
  assert(equalFloat(a.getVal(1, 2), 6));

  std::cout << "[PASS] basic get/set\n";

  // =========================
  // 2. 加法
  // =========================
  Matrix b{2, 3};

  b.setVal(0, 0, 10);
  b.setVal(0, 1, 20);
  b.setVal(0, 2, 30);
  b.setVal(1, 0, 40);
  b.setVal(1, 1, 50);
  b.setVal(1, 2, 60);

  Matrix add = a + b;

  assert(equalFloat(add.getVal(0, 0), 11));
  assert(equalFloat(add.getVal(0, 2), 33));
  assert(equalFloat(add.getVal(1, 2), 66));

  std::cout << "[PASS] addition\n";

  // =========================
  // 3. 减法
  // =========================
  Matrix sub = b - a;

  assert(equalFloat(sub.getVal(0, 0), 9));
  assert(equalFloat(sub.getVal(0, 2), 27));
  assert(equalFloat(sub.getVal(1, 2), 54));

  std::cout << "[PASS] subtraction\n";

  // =========================
  // 4. 标量乘法
  // =========================
  Matrix scalar = a * 2.0f;

  assert(equalFloat(scalar.getVal(0, 0), 2));
  assert(equalFloat(scalar.getVal(0, 2), 6));
  assert(equalFloat(scalar.getVal(1, 2), 12));

  std::cout << "[PASS] scalar multiplication\n";

  // =========================
  // 5. Hadamard 乘法
  // =========================
  Matrix had = a.hadamard(b);

  assert(equalFloat(had.getVal(0, 0), 10));
  assert(equalFloat(had.getVal(0, 1), 40));
  assert(equalFloat(had.getVal(1, 2), 360));

  std::cout << "[PASS] hadamard product\n";

  // =========================
  // 6. 转置
  // =========================
  Matrix trans = a.transpose();

  assert(trans.getRows() == 3);
  assert(trans.getCols() == 2);

  // 原矩阵：
  // 1 2 3
  // 4 5 6
  //
  // 转置：
  // 1 4
  // 2 5
  // 3 6

  assert(equalFloat(trans.getVal(0, 0), 1));
  assert(equalFloat(trans.getVal(0, 1), 4));
  assert(equalFloat(trans.getVal(1, 0), 2));
  assert(equalFloat(trans.getVal(2, 1), 6));

  std::cout << "[PASS] transpose\n";

  // =========================
  // 7. 矩阵乘法
  // =========================
  //
  // A: 2x3
  //
  // 1 2 3
  // 4 5 6
  //
  // C: 3x4
  //
  // 1  2  3  4
  // 5  6  7  8
  // 9 10 11 12
  //
  // A*C:
  //
  // 38  44  50  56
  // 83  98 113 128

  Matrix c{3, 4};

  c.setVal(0, 0, 1);
  c.setVal(0, 1, 2);
  c.setVal(0, 2, 3);
  c.setVal(0, 3, 4);

  c.setVal(1, 0, 5);
  c.setVal(1, 1, 6);
  c.setVal(1, 2, 7);
  c.setVal(1, 3, 8);

  c.setVal(2, 0, 9);
  c.setVal(2, 1, 10);
  c.setVal(2, 2, 11);
  c.setVal(2, 3, 12);

  Matrix mul = a * c;

  assert(mul.getRows() == 2);
  assert(mul.getCols() == 4);

  assert(equalFloat(mul.getVal(0, 0), 38));
  assert(equalFloat(mul.getVal(0, 1), 44));
  assert(equalFloat(mul.getVal(0, 2), 50));
  assert(equalFloat(mul.getVal(0, 3), 56));

  assert(equalFloat(mul.getVal(1, 0), 83));
  assert(equalFloat(mul.getVal(1, 1), 98));
  assert(equalFloat(mul.getVal(1, 2), 113));
  assert(equalFloat(mul.getVal(1, 3), 128));

  std::cout << "[PASS] matrix multiplication\n";

  std::cout << "\nAll Matrix tests passed!\n";

  return 0;
}