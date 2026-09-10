#include "recognizer.hpp"
#include "matrix.hpp"

Recognizer::Recognizer(const std::string &modelPath) : model(784, 10) {
  model.load(modelPath);
}

Prediction Recognizer::predict(const Matrix &input) {
  Matrix z = model.forward(input);
  z = this->softmaxLayer.forward(z);
  float confidence = z.getVal(0, 0);
  int predict_label{0};
  for (int i = 1; i < 10; i++) {
    if (z.getVal(i, 0) >= confidence) {
      confidence = z.getVal(i, 0);
      predict_label = i;
    }
  }
  return {predict_label, confidence};
}