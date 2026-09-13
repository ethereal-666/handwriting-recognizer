#include "cross_entropy.hpp"
#include "linear.hpp"
#include "matrix.hpp"
#include "mnist_loader.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>

#define SIZE 60000
#define LR 0.02f
#define HIDDEN_SIZE 256
#define EPOCH 10

int main() {
  // 训练
  // 创建文件读取器
  MNISTLoader mnist{"data/mnist/train-images-idx3-ubyte",
                    "data/mnist/train-labels-idx1-ubyte"};
  // 创建三层处理，linear->ReLU->linear
  linear linear1{784, HIDDEN_SIZE};
  ReLU relu;
  linear linear2{HIDDEN_SIZE, 10};
  softmax softmax{};
  CrossEntropy entropy{};
  float loss{0.0f};
  float sum_loss{0.0f};
  for (int time = 0; time < EPOCH; time++) {
    for (int num = 0; num < SIZE; num++) {
      // 读取一张图片，转化成784x1，归一化
      Matrix x0 = mnist.getImage(num).reshape(784, 1).normalization01();
      // 读取一个标签
      int label = mnist.getLabel(num);
      // 第一层线性计算
      Matrix x1 = linear1.forward(x0);
      // 第二层ReLU
      Matrix x2 = relu.forward(x1);
      // 第三层线性计算
      Matrix z = linear2.forward(x2);
      // softmax处理成概率
      z = softmax.forward(z);
      // 计算dz用于反向传播
      Matrix dz = entropy.backward(z, label);

      // 显示loss确认模型更新正确
      loss = entropy.forward(z, label);
      sum_loss += loss;
      if ((num + 1) % 1000 == 0) {
        std::cout << "EPOCH" << time + 1 << ":";
        std::cout << num - 999 << "~" << num << ":" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << sum_loss / 1000.0f
                  << std::endl;
        sum_loss = 0.0f;
      }

      // 反向传播
      Matrix dx = linear2.backward(dz);
      linear2.update(LR);
      dx = relu.backward(dx);
      linear1.backward(dx);
      linear1.update(LR);
    }
  }

  // 测试
  // 创建文件读取器
  MNISTLoader test{"data/mnist/t10k-images-idx3-ubyte",
                   "data/mnist/t10k-labels-idx1-ubyte"};

  int correct{0};

  for (int num = 0; num < 10000; num++) {
    // 读取一张图片，转化成784x1，归一化
    Matrix x0 = test.getImage(num).reshape(784, 1).normalization01();
    // 读取一个标签
    int label = test.getLabel(num);
    // 第一层线性计算
    Matrix x1 = linear1.forward(x0);
    // 第二层ReLU
    Matrix x2 = relu.forward(x1);
    // 第三层线性计算
    Matrix z = linear2.forward(x2);
    // softmax处理成概率
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

  linear1.setVersion(2);
  linear2.setVersion(2);
  linear1.save("models/mlp/l1v02.bin");
  linear2.save("models/mlp/l2v02.bin");
  return 0;
}