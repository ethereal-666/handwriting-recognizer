#include "cross_entropy.hpp"
#include "linear.hpp"
#include "matrix.hpp"
#include "mnist_loader.hpp"
#include "softmax.hpp"
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>

#define SIZE 60000

int main() {
  // 训练
  // 创建文件读取器
  MNISTLoader mnist{"data/mnist/train-images-idx3-ubyte",
                    "data/mnist/train-labels-idx1-ubyte"};
  // 创建linear模型
  linear model{784, 10};
  softmax softmax{};
  CrossEntropy entropy{};
  float loss{0.0f};
  float sum_loss{0.0f};

  for (int num = 0; num < SIZE; num++) {
    // 读取一张图片，转化成784x1，归一化
    Matrix x = mnist.getImage(num).reshape(784, 1).normalization01();
    // 读取一个标签
    int label = mnist.getLabel(num);
    // 预测计算，softmax处理，loss熵计算（用于输出查看），dz计算，计算dW、db，更新参数
    Matrix z = model.forward(x);
    z = softmax.forward(z);
    loss = entropy.forward(z, label);

    sum_loss += loss;
    if ((num + 1) % 1000 == 0) {
      std::cout << num - 999 << "~" << num << ":" << std::endl;
      std::cout << std::fixed << std::setprecision(6) << sum_loss / 1000.0f
                << std::endl;
      sum_loss = 0.0f;
    }

    Matrix dz = entropy.backward(z, label);
    model.backward(dz);
    model.update(0.002f);
  }

  // 测试
  // 创建文件读取器
  MNISTLoader test{"data/mnist/t10k-images-idx3-ubyte",
                   "data/mnist/t10k-labels-idx1-ubyte"};

  int correct{0};

  for (int num = 0; num < 10000; num++) {
    // 读取一张图片，转化成784x1，归一化
    Matrix x = test.getImage(num).reshape(784, 1).normalization01();
    // 读取一个标签
    int label = test.getLabel(num);
    // 预测计算，softmax处理，loss熵计算（用于输出查看），dz计算，计算dW、db，更新参数
    Matrix z = model.forward(x);
    z = softmax.forward(z);

    float tmp = z.getVal(0, 0);
    int index{0};
    for (int i = 1; i < 10; i++) {
      if (z.getVal(i, 0) >= tmp) {
        tmp = z.getVal(i, 0);
        index = i;
      }
    }
    if (index == label) {
      correct++;
    }
    if ((num + 1) % 1000 == 0) {
      std::cout << "test";
      std::cout << num - 999 << "~" << num << ":" << std::endl;
    }
  }
  std::cout << "正确率：" << std::fixed << std::setprecision(4)
            << (float)correct / 10000.0f << std::endl;

  model.setVersion(1);
  model.save("models/linear/modelv01.bin");

  return 0;
}