#pragma once
#include "linear.hpp"
#include "matrix.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include <string>
#include <vector>

struct Prediction {
  int label;
  float confidence;
};

class Recognizer {
private:
  std::vector<linear> models;
  softmax softmaxLayer{};
  ReLU reluLayer{};

public:
  Recognizer(const std::vector<std::string> &modelPaths); // 构造函数
  Prediction predict(const Matrix &input);                // 预测接口
};