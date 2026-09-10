#include "matrix.hpp"
#include "mnist_loader.hpp"
#include "recognizer.hpp"

#include <iomanip>
#include <iostream>

int main() {
  Recognizer recognizer{"models/modelv01.bin"};

  MNISTLoader testData{"data/mnist/t10k-images-idx3-ubyte",
                       "data/mnist/t10k-labels-idx1-ubyte"};

  int correct = 0;

  for (int i = 0; i < 10000; ++i) {
    Matrix input = testData.getImage(i).reshape(784, 1).normalization01();

    int realLabel = testData.getLabel(i);

    Prediction result = recognizer.predict(input);

    if (result.label == realLabel) {
      ++correct;
    }

    if (i < 10) {
      std::cout << "sample " << i << " | real=" << realLabel
                << " | predict=" << result.label
                << " | confidence=" << std::fixed << std::setprecision(4)
                << result.confidence * 100.0f << "%" << '\n';
    }

    if ((i + 1) % 1000 == 0) {
      std::cout << "progress: " << i + 1 << "/10000" << '\n';
    }
  }

  float accuracy = static_cast<float>(correct) / 10000.0f;

  std::cout << "\naccuracy: " << std::fixed << std::setprecision(4)
            << accuracy * 100.0f << "%" << '\n';

  return 0;
}