#include "linear.hpp"
#include "matrix.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <string>

linear::linear(int inputSize, int outputSize)
    : inputSize(inputSize), outputSize(outputSize), W(outputSize, inputSize),
      b(outputSize, 1), lastInput(inputSize, 1), dW(outputSize, inputSize),
      db(outputSize, 1) {
  // 随机初始化W
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  for (int i = 0; i < outputSize; i++) {
    for (int j = 0; j < inputSize; j++) {
      W.setVal(i, j, dist(gen));
    }
  }
}

Matrix linear::forward(const Matrix &input) {
  assert((input.getRows() == this->inputSize && input.getCols() == 1) &&
         "linear input size error");
  this->lastInput = input;
  Matrix result = W * input + b;
  return result;
}

Matrix linear::backward(const Matrix &dz) {
  assert((dz.getRows() == this->outputSize && dz.getCols() == 1) &&
         "dz size error");
  dW = dz * lastInput.transpose();
  db = dz;
  return W.transpose() * dz; // 返回dx
}

void linear::update(float learningRate) {
  assert((learningRate > 0) && "learning rate error");
  this->W = this->W - dW * learningRate;
  this->b = this->b - db * learningRate;
}

void linear::save(const std::string &path) const {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    std::cout << "open file failed" << std::endl;
    return;
  }

  // 写入magic
  std::string magic{"2006"};
  file.write(magic.data(), magic.size());
  // 写入version
  int version{1};
  file.write(reinterpret_cast<const char *>(&version), sizeof(version));
  // 写入inputsize、outputsize
  int tmp{this->inputSize};
  file.write(reinterpret_cast<const char *>(&tmp), sizeof(tmp));
  tmp = this->outputSize;
  file.write(reinterpret_cast<const char *>(&tmp), sizeof(tmp));
  // 写入W矩阵
  float value{0.0f};
  for (int i = 0; i < this->outputSize; i++) {
    for (int j = 0; j < this->inputSize; j++) {
      value = this->W.getVal(i, j);
      file.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
  }
  // 写入b矩阵
  for (int i = 0; i < this->outputSize; i++) {
    value = this->b.getVal(i, 0);
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }

  file.close();
}

void linear::load(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cout << "open file failed" << std::endl;
    return;
  }

  // 读取magic
  char magic[4];
  file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  std::string magicString{magic, 4};
  if (magicString != "2006") {
    std::cout << "model magic error" << std::endl;
    return;
  }
  // 读取version
  int version{};
  file.read(reinterpret_cast<char *>(&version), sizeof(version));
  // 读取inputsize、outputsize
  int tmp{this->inputSize};
  file.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
  this->inputSize = tmp;
  file.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
  this->outputSize = tmp;
  this->W = Matrix{this->outputSize, this->inputSize};
  this->b = Matrix{this->outputSize, 1};
  this->lastInput = Matrix{this->inputSize, 1};
  this->dW = Matrix{this->outputSize, this->inputSize};
  this->db = Matrix{this->outputSize, 1};
  // 读取W矩阵
  float value{0.0f};
  for (int i = 0; i < this->outputSize; i++) {
    for (int j = 0; j < this->inputSize; j++) {
      file.read(reinterpret_cast<char *>(&value), sizeof(value));
      this->W.setVal(i, j, value);
    }
  }
  // 读取b矩阵
  for (int i = 0; i < this->outputSize; i++) {
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    this->b.setVal(i, 0, value);
  }

  file.close();
}