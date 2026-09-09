#include "matrix.hpp"
#include <cassert>
#include <iostream>

Matrix::Matrix(int rows, int cols) {
  assert((rows > 0 && cols > 0) && "Matrix index error");
  this->rows = rows;
  this->cols = cols;
  this->data.resize(rows * cols);
}

int Matrix::getRows() const {
  return this->rows;
}

int Matrix::getCols() const {
  return this->cols;
}

float Matrix::getVal(int row, int col) const {
  assert((row >= 0 && col >= 0 && row < this->rows && col < this->cols) &&
         "Matrix getVal index error");
  return data[row * this->cols + col];
}

void Matrix::setVal(int row, int col, float value) {
  assert((row >= 0 && col >= 0 && row < this->rows && col < this->cols) &&
         "Matrix setVal error");
  data[row * this->cols + col] = value;
}

void Matrix::printMatrix() const {
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      std::cout << data[i * this->cols + j];
      std::cout << ' ';
    }
    std::cout << std::endl;
  }
}

Matrix Matrix::operator+(const Matrix &a) const {
  assert((this->rows == a.rows && this->cols == a.cols) && "Matrix + error");
  Matrix result{this->rows, this->cols};
  for (int i = 0; i < (this->rows * this->cols); i++) {
    result.data[i] = this->data[i] + a.data[i];
  }
  return result;
}

Matrix Matrix::operator-(const Matrix &a) const {
  assert((this->rows == a.rows && this->cols == a.cols) && "Matrix - error");
  Matrix result{this->rows, this->cols};
  for (int i = 0; i < (this->rows * this->cols); i++) {
    result.data[i] = this->data[i] - a.data[i];
  }
  return result;
}

Matrix Matrix::operator*(const Matrix &a) const {
  assert((this->cols == a.rows) && "Matrix * error");
  Matrix result{this->rows, a.cols};
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      result.data[i * a.cols + j] = 0;
      for (int k = 0; k < this->cols; k++) {
        result.data[i * a.cols + j] +=
            this->data[i * cols + k] * a.data[k * a.cols + j];
      }
    }
  }
  return result;
}

Matrix Matrix::hadamard(const Matrix &a) const {
  assert((this->rows == a.rows && this->cols == a.cols) &&
         "Matrix hadamard error");
  Matrix result{this->rows, this->cols};
  for (int i = 0; i < (this->rows * this->cols); i++) {
    result.data[i] = this->data[i] * a.data[i];
  }
  return result;
}

Matrix Matrix::operator*(float scalar) const {
  Matrix result{this->rows, this->cols};
  for (int i = 0; i < (this->rows * this->cols); i++) {
    result.data[i] = scalar * this->data[i];
  }
  return result;
}

Matrix Matrix::transpose() const {
  Matrix result{this->cols, this->rows};
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      result.data[j * result.cols + i] = this->data[i * this->cols + j];
    }
  }
  return result;
}