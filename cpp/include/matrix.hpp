#pragma once
#include <vector>

class Matrix { // 矩阵类
private:
  // 行、列、值
  int rows;
  int cols;
  std::vector<float> data;

public:
  Matrix(int rows, int cols);                     // 构造函数
  int getRows() const;                            // 查询行数
  int getCols() const;                            // 查询列数
  float getVal(int row, int col) const;           // 根据查询值
  void setVal(int row, int col, float value);     // 设置值
  void printMatrix() const;                       // 打印整个矩阵
  Matrix operator+(const Matrix &a) const;        // 矩阵加法
  Matrix operator-(const Matrix &a) const;        // 矩阵减法
  Matrix operator*(const Matrix &a) const;        // 矩阵乘法
  Matrix hadamard(const Matrix &a) const;         // 哈达玛积
  Matrix operator*(float scalar) const;           // 标量乘法
  Matrix transpose() const;                       // 矩阵转置
  Matrix reshape(int newRows, int newCols) const; // 改变行列，实际不动底层
  Matrix normalization01() const;
};