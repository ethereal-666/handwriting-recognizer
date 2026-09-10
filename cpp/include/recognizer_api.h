// 对python提供C接口

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int label;
  float confidence;
} PredictionResult;

void *recognizer_create(const char *modelPath);

PredictionResult recognizer_predict(void *handle, const float *input, int size);

void recognizer_destroy(void *handle);

#ifdef __cplusplus
}
#endif