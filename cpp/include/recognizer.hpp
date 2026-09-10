#pragma once
#include "linear.hpp"
#include "matrix.hpp"
#include "softmax.hpp"
#include <string>

struct Prediction {
  int label;
  float confidence;
};

class Recognizer {
private:
  linear model;
  softmax softmaxLayer{};

public:
  Recognizer(const std::string &modelPath); // 构造函数
  Prediction predict(const Matrix &input);  // 预测接口
};