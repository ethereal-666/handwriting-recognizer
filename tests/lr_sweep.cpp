#include "cross_entropy.hpp"
#include "linear.hpp"
#include "matrix.hpp"
#include "mnist_loader.hpp"
#include "softmax.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

constexpr int ROUGH_TRAIN_SIZE = 20000;
constexpr int FULL_TRAIN_SIZE = 60000;
constexpr int TEST_SIZE = 10000;

constexpr int LR_COUNT = 32;
constexpr int ROUGH_TOP_K = 4;

struct Result {
  float learningRate;
  float accuracy;
};

std::mutex coutMutex;

int argmax(const Matrix &x) {
  float maxValue = x.getVal(0, 0);
  int maxIndex = 0;

  for (int i = 1; i < x.getRows(); ++i) {
    float value = x.getVal(i, 0);

    if (value > maxValue) {
      maxValue = value;
      maxIndex = i;
    }
  }

  return maxIndex;
}

void printProgress(const std::string &stage, int workerId, float lr,
                   int current, int total) {
  std::lock_guard<std::mutex> lock(coutMutex);

  std::cout << "[" << stage << "] worker " << std::setw(2) << workerId
            << " | lr=" << std::fixed << std::setprecision(6) << lr
            << " | progress " << current << "/" << total << std::endl;
}

float evaluate(float learningRate, int trainSize, int workerId,
               const std::string &stageName) {
  MNISTLoader trainData{"data/mnist/train-images-idx3-ubyte",
                        "data/mnist/train-labels-idx1-ubyte"};

  MNISTLoader testData{"data/mnist/t10k-images-idx3-ubyte",
                       "data/mnist/t10k-labels-idx1-ubyte"};

  linear model{784, 10};
  softmax softmaxLayer{};
  CrossEntropy entropy{};

  // =========================
  // train
  // =========================
  for (int i = 0; i < trainSize; ++i) {
    Matrix x = trainData.getImage(i).reshape(784, 1).normalization01();

    int label = trainData.getLabel(i);

    Matrix z = model.forward(x);
    Matrix probabilities = softmaxLayer.forward(z);

    Matrix dz = entropy.backward(probabilities, label);

    model.backward(dz);
    model.update(learningRate);

    if ((i + 1) % 2000 == 0 || i + 1 == trainSize) {
      printProgress(stageName, workerId, learningRate, i + 1, trainSize);
    }
  }

  // =========================
  // test
  // =========================
  int correct = 0;

  for (int i = 0; i < TEST_SIZE; ++i) {
    Matrix x = testData.getImage(i).reshape(784, 1).normalization01();

    int label = testData.getLabel(i);

    Matrix z = model.forward(x);
    Matrix probabilities = softmaxLayer.forward(z);

    int prediction = argmax(probabilities);

    if (prediction == label) {
      ++correct;
    }

    if ((i + 1) % 2000 == 0 || i + 1 == TEST_SIZE) {
      printProgress(stageName + "-test", workerId, learningRate, i + 1,
                    TEST_SIZE);
    }
  }

  return static_cast<float>(correct) / static_cast<float>(TEST_SIZE);
}

int main() {
  // 围绕你已经试出来比较好的区间，细化搜索
  std::vector<float> learningRates{
      0.00050f, 0.00055f, 0.00060f, 0.00065f, 0.00070f, 0.00075f, 0.00080f,
      0.00085f, 0.00090f, 0.00095f, 0.00100f, 0.00105f, 0.00110f, 0.00115f,
      0.00120f, 0.00125f, 0.00130f, 0.00135f, 0.00140f, 0.00145f, 0.00150f,
      0.00155f, 0.00160f, 0.00165f, 0.00170f, 0.00175f, 0.00180f, 0.00185f,
      0.00190f, 0.00195f, 0.00200f, 0.00205f};

  std::vector<Result> roughResults(LR_COUNT);
  std::vector<std::thread> threads;

  // =========================
  // Stage 1: rough search
  // =========================
  {
    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << "========== Stage 1: Rough Search ==========\n";
  }

  threads.reserve(LR_COUNT);

  for (int i = 0; i < LR_COUNT; ++i) {
    threads.emplace_back([&, i]() {
      float lr = learningRates[i];

      float accuracy = evaluate(lr, ROUGH_TRAIN_SIZE, i, "rough");

      roughResults[i] = {lr, accuracy};

      std::lock_guard<std::mutex> lock(coutMutex);

      std::cout << "[rough-done] worker " << std::setw(2) << i
                << " | lr=" << std::fixed << std::setprecision(6) << lr
                << " | accuracy=" << std::setprecision(4) << accuracy
                << std::endl;
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  std::sort(
      roughResults.begin(), roughResults.end(),
      [](const Result &a, const Result &b) { return a.accuracy > b.accuracy; });

  {
    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << "\n========== Stage 1 Top 4 ==========\n";

    for (int i = 0; i < ROUGH_TOP_K; ++i) {
      std::cout << "rank " << i + 1 << " | lr=" << std::fixed
                << std::setprecision(6) << roughResults[i].learningRate
                << " | accuracy=" << std::setprecision(4)
                << roughResults[i].accuracy << std::endl;
    }
  }

  // =========================
  // Stage 2: full training
  // =========================
  std::vector<Result> finalResults(ROUGH_TOP_K);
  threads.clear();

  {
    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << "\n========== Stage 2: Full Training ==========\n";
  }

  for (int i = 0; i < ROUGH_TOP_K; ++i) {
    threads.emplace_back([&, i]() {
      float lr = roughResults[i].learningRate;

      float accuracy = evaluate(lr, FULL_TRAIN_SIZE, i, "full");

      finalResults[i] = {lr, accuracy};

      std::lock_guard<std::mutex> lock(coutMutex);

      std::cout << "[full-done] worker " << std::setw(2) << i
                << " | lr=" << std::fixed << std::setprecision(6) << lr
                << " | final accuracy=" << std::setprecision(4) << accuracy
                << std::endl;
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  std::sort(
      finalResults.begin(), finalResults.end(),
      [](const Result &a, const Result &b) { return a.accuracy > b.accuracy; });

  {
    std::lock_guard<std::mutex> lock(coutMutex);

    std::cout << "\n========== Final Result ==========\n";

    for (int i = 0; i < ROUGH_TOP_K; ++i) {
      std::cout << "rank " << i + 1 << " | lr=" << std::fixed
                << std::setprecision(6) << finalResults[i].learningRate
                << " | accuracy=" << std::setprecision(4)
                << finalResults[i].accuracy << std::endl;
    }

    std::cout << "\nBest learning rate: " << std::fixed << std::setprecision(6)
              << finalResults[0].learningRate << std::endl;
  }

  return 0;
}