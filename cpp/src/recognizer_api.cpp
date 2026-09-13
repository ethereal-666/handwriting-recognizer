#include "recognizer_api.h"
#include "matrix.hpp"
#include "recognizer.hpp"
#include <string>
#include <vector>

void *recognizer_create(const char *modelPath[], int num) {
  std::vector<std::string> modelpaths;
  for (int i = 0; i < num; i++) {
    modelpaths.push_back(modelPath[i]);
  }
  Recognizer *model = new Recognizer(modelpaths); // 要长期保留一直到结束，要new
  return model;
}

PredictionResult recognizer_predict(void *handle, const float *input,
                                    int size) {
  if (!handle || !input || size != 784) {
    return {-1, 0.0f};
  }
  Recognizer *model = static_cast<Recognizer *>(handle);
  Matrix x{784, 1};
  for (int i = 0; i < 784; i++) {
    x.setVal(i, 0, input[i]);
  }
  Prediction result = model->predict(x);
  return {result.label, result.confidence};
}

void recognizer_destroy(void *handle) {
  Recognizer *model = static_cast<Recognizer *>(handle);
  delete model;
}