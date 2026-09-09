#include "mnist_loader.hpp"

#include <cassert>
#include <iostream>

int main() {
  MNISTLoader loader{"data/mnist/train-images-idx3-ubyte",
                     "data/mnist/train-labels-idx1-ubyte"};

  // 1. 测试文件头
  assert(loader.getImageCount() == 60000);
  assert(loader.getLabelCount() == 60000);
  assert(loader.getRows() == 28);
  assert(loader.getCols() == 28);

  std::cout << "[PASS] MNIST header\n";

  // 2. 读取第一张图片和标签
  Matrix image = loader.getImage(0);
  int label = loader.getLabel(0);

  assert(image.getRows() == 28);
  assert(image.getCols() == 28);
  assert(label >= 0 && label <= 9);

  std::cout << "[PASS] read first image\n";
  std::cout << "label = " << label << '\n';

  // 标准 MNIST 训练集第一张应该是 5
  assert(label == 5);

  // 3. 在终端把图片画出来
  for (int i = 0; i < image.getRows(); ++i) {
    for (int j = 0; j < image.getCols(); ++j) {
      float pixel = image.getVal(i, j);

      if (pixel > 128) {
        std::cout << "##";
      } else if (pixel > 30) {
        std::cout << "++";
      } else {
        std::cout << "  ";
      }
    }
    std::cout << '\n';
  }

  std::cout << "[PASS] MNIST loader\n";

  return 0;
}