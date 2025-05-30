#ifndef DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
#define DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

struct CAMParams
{
  int orig_h;
  int orig_w;
  int network_h;
  int network_w;
  double fx;
  double fy;
  double cx;
  double cy;
};

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity <= Severity::kINFO)
      std::cout << "[TRT] " << msg << std::endl;
  }
};

class MonoDepthEstimation
{
public:
  MonoDepthEstimation(const CAMParams &, const std::string &);
  ~MonoDepthEstimation();

  void runInference(const cv::Mat &input_img);

  cv::Mat depth_img_;
  sensor_msgs::msg::PointCloud2 depth_cloud_;

private:
  int resize_h_, resize_w_;
  Logger gLogger;
  std::vector<float> result_;
  float fx_, fy_, cx_, cy_;
  bool use_rgb_ = true;
  float scaled_fx_, scaled_fy_;
  float scaled_cx_, scaled_cy_;
  const float min_depth_ = 0.1f;
  const float max_depth_ = 100.0f;
  float min_disp_, max_disp_;
  const float STEREO_SCALE_FACTOR = 5.4f;

  // Buffers
  void *buffers_[2];
  float *input_host_ = nullptr;
  float *output_host_ = nullptr;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream_;

  cv::Mat preprocessImage(const cv::Mat &, int, int);
  std::vector<float> imageToTensor(const cv::Mat &);
  void initializeTRT(const std::string &);
  cv::Mat normalizeRGB(const cv::Mat &input);
  cv::Mat computeDepth();
  cv::Mat convertToDepthImg(const cv::Mat &);
  void initializeDepthCloud();
  void createPointCloudFromDepth(const cv::Mat &, const cv::Mat &);
};

#endif // DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
