#ifndef DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
#define DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kINFO)
      std::cout << "[TRT] " << msg << std::endl;
  }
};

class MonoDepthEstimation {
public:
  MonoDepthEstimation(int, int, const std::string &);

  // Delete copy constructor and assignment
  MonoDepthEstimation(const MonoDepthEstimation &) = delete;
  MonoDepthEstimation &operator=(const MonoDepthEstimation &) = delete;

  // Allow move semantics
  MonoDepthEstimation(MonoDepthEstimation &&) noexcept = default;
  MonoDepthEstimation &operator=(MonoDepthEstimation &&) noexcept = default;

  void runInference(const cv::Mat &input_img);

  cv::Mat depth_img_;

private:
  int resize_h_, resize_w_;
  cv::Mat depth_map_;
  Logger gLogger;
  float MAX_DEPTH = 80.0f;
  std::vector<float> result_;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream_;

  cv::Mat preprocessImage(const cv::Mat &, int, int);
  std::vector<float> imageToTensor(const cv::Mat &);
  void initializeTRT(const std::string &);

  cv::Mat convertToDepthMap();
  cv::Mat convertToDepthImg();
};

#endif // DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
