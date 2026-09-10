#pragma once
#include "matrix.hpp"
#include <fstream>
#include <string>

class MNISTLoader {
private:
  // 图片文件路径
  std::string imagePath;
  // 标签文件路径
  std::string labelPath;
  // image对应file
  std::ifstream imageFile;
  // label对应file
  std::ifstream labelFile;

  // 图片数量
  int imageCount;
  // 标签数量
  int labelCount;

  // 图片行数
  int rows;
  // 图片列数
  int cols;

  int readBigEndianInt(std::ifstream &file) const; // 大端转小端读取int

public:
  MNISTLoader(const std::string &imagePath,
              const std::string &labelPath); // 构造函数
  int getImageCount() const;                 // 查询图片数量
  int getLabelCount() const;                 // 查询标签数量
  int getRows() const;                       // 查询图片行数
  int getCols() const;                       // 查询图片列数
  Matrix getImage(int index);                // 得到指定位置图片信息
  int getLabel(int index);                   // 得到指定位置图片标签
};