#include "recognizer.hpp"
#include "linear.hpp"
#include "matrix.hpp"

Recognizer::Recognizer(const std::vector<std::string> &modelPaths) {
  for (int i = 0; i < modelPaths.size(); i++) {
    linear l{1, 1};
    l.load(modelPaths[i]);
    this->models.push_back(l);
  }
}

Prediction Recognizer::predict(const Matrix &input) {
  Matrix z{784, 1};

  switch (this->models.size()) {
  case 1:
    z = models[0].forward(input);
    break;
  case 2:
    z = models[0].forward(input);
    z = this->reluLayer.forward(z);
    z = models[1].forward(z);
    break;
  }

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