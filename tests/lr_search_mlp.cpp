#include "cross_entropy.hpp"
#include "linear.hpp"
#include "matrix.hpp"
#include "mnist_loader.hpp"
#include "relu.hpp"
#include "softmax.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define HIDDEN_SIZE 256

// 搜索学习率时不用一上来训练 5 epoch
#define SEARCH_EPOCH 6

// 最多使用 32 个线程
#define MAX_THREADS 32

struct TrialResult {
  float learningRate{0.0f};
  float accuracy{0.0f};
  float loss{0.0f};
  bool valid{true};
  double seconds{0.0};
};

std::mutex printMutex;

TrialResult runTrial(float learningRate, const std::string &initL1Path,
                     const std::string &initL2Path) {
  auto start = std::chrono::steady_clock::now();

  TrialResult result;
  result.learningRate = learningRate;

  // 每个线程必须拥有自己的 Loader
  // 不要多个线程共享同一个 ifstream
  MNISTLoader mnist{"data/mnist/train-images-idx3-ubyte",
                    "data/mnist/train-labels-idx1-ubyte"};

  // 每个线程也拥有自己独立的模型
  linear linear1{784, HIDDEN_SIZE};
  ReLU relu;
  linear linear2{HIDDEN_SIZE, 10};
  softmax softmax{};
  CrossEntropy entropy{};

  // 所有学习率从相同的初始权重开始
  linear1.load(initL1Path);
  linear2.load(initL2Path);

  float finalLoss{0.0f};

  for (int epoch = 0; epoch < SEARCH_EPOCH; epoch++) {
    float sumLoss{0.0f};

    for (int num = 0; num < TRAIN_SIZE; num++) {
      Matrix x0 = mnist.getImage(num).reshape(784, 1).normalization01();

      int label = mnist.getLabel(num);

      // ---------- forward ----------
      Matrix x1 = linear1.forward(x0);
      Matrix x2 = relu.forward(x1);
      Matrix z = linear2.forward(x2);
      Matrix probability = softmax.forward(z);

      float loss = entropy.forward(probability, label);

      // 如果学习率太大导致数值爆炸，直接结束这个实验
      if (!std::isfinite(loss)) {
        result.valid = false;
        result.loss = loss;

        auto end = std::chrono::steady_clock::now();
        result.seconds = std::chrono::duration<double>(end - start).count();

        return result;
      }

      sumLoss += loss;

      // ---------- backward ----------
      Matrix dz = entropy.backward(probability, label);

      Matrix dx = linear2.backward(dz);
      linear2.update(learningRate);

      dx = relu.backward(dx);

      linear1.backward(dx);
      linear1.update(learningRate);
    }

    finalLoss = sumLoss / static_cast<float>(TRAIN_SIZE);
  }

  result.loss = finalLoss;

  // ---------- 测试 ----------
  MNISTLoader test{"data/mnist/t10k-images-idx3-ubyte",
                   "data/mnist/t10k-labels-idx1-ubyte"};

  int correct{0};

  for (int num = 0; num < TEST_SIZE; num++) {
    Matrix x0 = test.getImage(num).reshape(784, 1).normalization01();

    int label = test.getLabel(num);

    Matrix x1 = linear1.forward(x0);
    Matrix x2 = relu.forward(x1);
    Matrix z = linear2.forward(x2);
    Matrix probability = softmax.forward(z);

    float maxProbability = probability.getVal(0, 0);
    int prediction{0};

    for (int i = 1; i < 10; i++) {
      float p = probability.getVal(i, 0);

      if (p > maxProbability) {
        maxProbability = p;
        prediction = i;
      }
    }

    if (prediction == label) {
      correct++;
    }
  }

  result.accuracy = static_cast<float>(correct) / static_cast<float>(TEST_SIZE);

  auto end = std::chrono::steady_clock::now();

  result.seconds = std::chrono::duration<double>(end - start).count();

  return result;
}

int main() {
  /*
   * 你已经发现 0.2 效果不错，所以重点搜索它附近。
   *
   * 共 32 个学习率，可以最多同时跑 32 个线程。
   */
  const std::vector<float> learningRates{
      0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
      0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f,
      0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f,
      0.30f, 0.32f, 0.35f, 0.38f, 0.42f, 0.46f, 0.50f, 0.60f};

  /*
   * 创建统一的初始权重。
   *
   * 如果直接让每个线程自己 new linear，
   * 每个学习率的随机初始权重不一样，
   * 比较结果就不公平。
   */
  const std::string initL1Path{"tests/.lr_init_l1.bin"};
  const std::string initL2Path{"tests/.lr_init_l2.bin"};

  {
    linear initialLinear1{784, HIDDEN_SIZE};
    linear initialLinear2{HIDDEN_SIZE, 10};

    initialLinear1.save(initL1Path);
    initialLinear2.save(initL2Path);
  }

  std::vector<TrialResult> results(learningRates.size());

  /*
   * atomic index 相当于任务队列。
   *
   * 每个线程：
   *
   * nextIndex.fetch_add(1)
   *          ↓
   * 获取下一个学习率
   *          ↓
   * 独立训练
   */
  std::atomic<std::size_t> nextIndex{0};

  unsigned int hardwareThreads = std::thread::hardware_concurrency();

  if (hardwareThreads == 0) {
    hardwareThreads = MAX_THREADS;
  }

  unsigned int threadCount = std::min<unsigned int>(
      MAX_THREADS,
      std::min<unsigned int>(hardwareThreads,
                             static_cast<unsigned int>(learningRates.size())));

  std::cout << "Hardware threads: " << hardwareThreads << '\n';

  std::cout << "Worker threads: " << threadCount << '\n';

  std::cout << "Learning rate candidates: " << learningRates.size() << '\n';

  std::cout << "Epoch per candidate: " << SEARCH_EPOCH << "\n\n";

  auto totalStart = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(threadCount);

  for (unsigned int threadId = 0; threadId < threadCount; threadId++) {

    workers.emplace_back([&]() {
      while (true) {
        std::size_t index = nextIndex.fetch_add(1);

        if (index >= learningRates.size()) {
          break;
        }

        float lr = learningRates[index];

        {
          std::lock_guard<std::mutex> lock(printMutex);

          std::cout << "[START] lr = " << std::fixed << std::setprecision(3)
                    << lr << '\n';
        }

        TrialResult trial = runTrial(lr, initL1Path, initL2Path);

        results[index] = trial;

        {
          std::lock_guard<std::mutex> lock(printMutex);

          if (trial.valid) {
            std::cout << "[DONE ] lr = " << std::fixed << std::setprecision(3)
                      << trial.learningRate
                      << "  accuracy = " << std::setprecision(4)
                      << trial.accuracy << "  loss = " << std::setprecision(6)
                      << trial.loss << "  time = " << std::setprecision(2)
                      << trial.seconds << "s\n";
          } else {
            std::cout << "[FAIL ] lr = " << trial.learningRate
                      << " numerical explosion\n";
          }
        }
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  auto totalEnd = std::chrono::steady_clock::now();

  double totalSeconds =
      std::chrono::duration<double>(totalEnd - totalStart).count();

  // 删除临时初始化模型
  std::remove(initL1Path.c_str());
  std::remove(initL2Path.c_str());

  // 排名
  std::vector<TrialResult> ranking = results;

  std::sort(ranking.begin(), ranking.end(),
            [](const TrialResult &a, const TrialResult &b) {
              if (a.valid != b.valid) {
                return a.valid > b.valid;
              }

              if (a.accuracy != b.accuracy) {
                return a.accuracy > b.accuracy;
              }

              // accuracy 相同的时候，loss 更低的优先
              return a.loss < b.loss;
            });

  std::cout << "\n========================================\n";
  std::cout << "Learning Rate Ranking\n";
  std::cout << "========================================\n";

  std::cout << std::left << std::setw(8) << "Rank" << std::setw(12) << "LR"
            << std::setw(14) << "Accuracy" << std::setw(14) << "Loss"
            << std::setw(12) << "Time" << '\n';

  int rank{1};

  for (const auto &trial : ranking) {
    if (!trial.valid) {
      continue;
    }

    std::cout << std::left << std::setw(8) << rank++ << std::setw(12)
              << std::fixed << std::setprecision(3) << trial.learningRate

              << std::setw(14) << std::setprecision(4) << trial.accuracy

              << std::setw(14) << std::setprecision(6) << trial.loss

              << std::setw(12) << std::setprecision(2) << trial.seconds << '\n';
  }

  std::cout << "\nTotal time: " << std::fixed << std::setprecision(2)
            << totalSeconds << "s\n";

  if (!ranking.empty() && ranking[0].valid) {
    std::cout << "\nBest learning rate: " << ranking[0].learningRate
              << "\nBest accuracy: " << ranking[0].accuracy << '\n';
  }

  return 0;
}