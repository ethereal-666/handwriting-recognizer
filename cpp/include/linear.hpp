#pragma once
#include "matrix.hpp"
#include <string>
class linear {
private:
  int version;      // 模型版本
  int inputSize;    // 输入规模
  int outputSize;   // 输出规模
  Matrix W;         // 权重矩阵
  Matrix b;         // 偏置矩阵
  Matrix lastInput; // 上一次输入，用于backward中dW计算
  Matrix dW;        // 用于更新参数W
  Matrix db;        // 用于更新参数b

public:
  void setVersion(int version);             // 设置版本
  linear(int inputSize, int outputSize);    // 构造函数
  Matrix forward(const Matrix &input);      // 前向计算
  Matrix backward(const Matrix &dz);        // 算dW、db、dx,只返回dx用于其他层
  void update(float learningRate);          // 更新参数
  void save(const std::string &path) const; // 保存模型
  void load(const std::string &path);       // 加载模型
};