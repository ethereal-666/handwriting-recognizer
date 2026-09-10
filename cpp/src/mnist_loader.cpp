#include "mnist_loader.hpp"
#include <cassert>
#include <fstream>

MNISTLoader::MNISTLoader(const std::string &imagePath,
                         const std::string &labelPath)
    : imagePath(imagePath), labelPath(labelPath) {

  // 打开图片和标签文件并校验
  this->imageFile.open(imagePath, std::ios::binary);
  this->labelFile.open(labelPath, std::ios::binary);
  assert(this->imageFile && "imageFile open error");
  assert(this->labelFile && "labelFile open error");
  int imageMagic = readBigEndianInt(this->imageFile);
  assert((imageMagic == 0x803) && "invalid imageFile file(magic number error)");
  int labelMagic = readBigEndianInt(this->labelFile);
  assert((labelMagic == 0x801) && "invalid labelFile file(magic number error)");
  // 读取图片和标签参数
  this->imageCount = readBigEndianInt(this->imageFile);
  this->rows = readBigEndianInt(this->imageFile);
  this->cols = readBigEndianInt(this->imageFile);
  this->labelCount = readBigEndianInt(this->labelFile);
}

int MNISTLoader::readBigEndianInt(std::ifstream &file) const {
  int x{0};
  file.read(reinterpret_cast<char *>(&x), 4);
  return ((x & 0xFF000000) >> 24) | ((x & 0x00FF0000) >> 8) |
         ((x & 0x0000FF00) << 8) | ((x & 0x000000FF) << 24);
}

int MNISTLoader::getImageCount() const {
  return this->imageCount;
}

int MNISTLoader::getLabelCount() const {
  return this->labelCount;
}

int MNISTLoader::getRows() const {
  return this->rows;
}

int MNISTLoader::getCols() const {
  return this->cols;
}

Matrix MNISTLoader::getImage(int index) {
  assert((index >= 0 && (index < this->imageCount)) && "invalid index");
  int offset{16 + index * this->rows * this->cols};
  this->imageFile.seekg(offset, std::ios::beg);
  Matrix result{this->rows, this->cols};
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      unsigned char tmp{0};
      imageFile.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
      result.setVal(i, j, static_cast<float>(tmp));
    }
  }
  return result;
}

int MNISTLoader::getLabel(int index) {
  assert((index >= 0 && (index < this->labelCount)) && "invalid index");
  int offset{8 + index};
  this->labelFile.seekg(offset, std::ios::beg);
  unsigned char label{0};
  this->labelFile.read(reinterpret_cast<char *>(&label), 1);
  assert((label >= 0 && label <= 9) && "read label error");
  return static_cast<int>(label);
}