#include "mono_depth/depth_estimation.hpp"
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>

MonoDepthEstimation::MonoDepthEstimation(const CAMParams &cam_params,
                                         const std::string &depth_weight_file)
{
  // Set depth img size and detection img size
  resize_h_ = cam_params.network_h;
  resize_w_ = cam_params.network_w;

  // Calcualte scaled cam intrinsics
  float scale_x = static_cast<float>(cam_params.network_w) / cam_params.orig_w;
  float scale_y = static_cast<float>(cam_params.network_h) / cam_params.orig_h;

  scaled_fx_ = static_cast<float>(cam_params.fx) * scale_x;
  scaled_fy_ = static_cast<float>(cam_params.fy) * scale_y;
  scaled_cx_ = static_cast<float>(cam_params.cx) * scale_x;
  scaled_cy_ = static_cast<float>(cam_params.cy) * scale_y;

  // Calculate min and max disparity for depth calculation
  min_disp_ = 1.0f / max_depth_;
  max_disp_ = 1.0f / min_depth_;

  // initialize depth cloud
  initializeDepthCloud();

  // Set up TRT
  initializeTRT(depth_weight_file);

  // Allocate buffers
  cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&input_host_),
                                   1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[0], 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_host_),
                 resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[1], resize_h_ * resize_w_ * sizeof(float));

  // Create stream
  cudaStreamCreate(&stream_);
}

MonoDepthEstimation::~MonoDepthEstimation()
{
  if(buffers_[0])
  {
    cudaFree(buffers_[0]);
    buffers_[0] = nullptr;
  }
  if(buffers_[1])
  {
    cudaFree(buffers_[1]);
    buffers_[1] = nullptr;
  }
  if(input_host_)
  {
    cudaFreeHost(input_host_);
    input_host_ = nullptr;
  }
  if(output_host_)
  {
    cudaFreeHost(output_host_);
    output_host_ = nullptr;
  }
  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}

cv::Mat MonoDepthEstimation::normalizeRGB(const cv::Mat &input)
{
  std::vector<cv::Mat> channels(3);
  cv::split(input, channels);

  std::vector<cv::Mat> temp_data;
  temp_data.resize(3);

  for(int i = 0; i < 3; ++i)
  {
    cv::Mat float_channel;
    channels[i].convertTo(float_channel, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(float_channel, mean, stddev);

    // Normalize: (x - mean) / std
    temp_data[i] = (float_channel - mean[0]) / stddev[0];
  }

  // Convert to cv::Mat
  cv::Mat normalized_rgb;
  cv::vconcat(temp_data, normalized_rgb);

  return normalized_rgb;
}

cv::Mat MonoDepthEstimation::preprocessImage(const cv::Mat &image, int input_width,
                                             int input_height)
{
  cv::Mat resized, chw_image;

  // Resize to model input size
  cv::resize(image, resized, cv::Size(input_width, input_height));

  // Convert to float32 and CHW
  chw_image = normalizeRGB(resized);

  return chw_image;
}

std::vector<float> MonoDepthEstimation::imageToTensor(const cv::Mat &mat)
{
  std::vector<float> tensor_data;
  if(mat.isContinuous())
    tensor_data.assign((float *)mat.datastart, (float *)mat.dataend);
  else
  {
    // Convert from HWC to CHW
    if(mat.channels() == 1)
    {
      // Single-channel (grayscale)
      for(int i = 0; i < mat.rows; ++i)
      {
        const float *row_ptr = mat.ptr<float>(i);
        tensor_data.insert(tensor_data.end(), row_ptr, row_ptr + mat.cols);
      }
    }
    else
    {
      // Multi-channel (e.g., RGB = 3 channels)
      for(int c = 0; c < mat.channels(); ++c)
      {
        for(int i = 0; i < mat.rows; ++i)
        {
          for(int j = 0; j < mat.cols; ++j)
          {
            const cv::Vec<float, 3> &pixel = mat.at<cv::Vec<float, 3>>(i, j);
            tensor_data.push_back(pixel[c]);
          }
        }
      }
    }
  }
  return tensor_data;
}

void MonoDepthEstimation::initializeTRT(const std::string &engine_file)
{
  // Load TensorRT engine from file
  std::ifstream file(engine_file, std::ios::binary);
  if(!file)
  {
    throw std::runtime_error("Failed to open engine file: " + engine_file);
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);

  // Create runtime and deserialize engine
  // Create TensorRT Runtime
  runtime.reset(nvinfer1::createInferRuntime(gLogger));

  // Deserialize engine
  engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  context.reset(engine->createExecutionContext());
}

void MonoDepthEstimation::runInference(const cv::Mat &input_img)
{
  // Preprocess image and convert to vector
  cv::Mat processed_img = preprocessImage(input_img, resize_w_, resize_h_);
  std::vector<float> input_tensor = imageToTensor(processed_img);

  // Copy to host memory and then to GPU
  std::memcpy(input_host_, input_tensor.data(),
              1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMemcpyAsync(buffers_[0], input_host_, 1 * 3 * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Set up inference buffers
  context->setInputTensorAddress("input", buffers_[0]);
  context->setOutputTensorAddress("depth", buffers_[1]);

  // inference
  context->enqueueV3(stream_);

  // Copy the result back
  cudaMemcpyAsync(output_host_, buffers_[1], resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  cudaStreamSynchronize(stream_);

  int output_size = resize_h_ * resize_w_;
  result_.assign(output_host_, output_host_ + output_size);

  // Convert to cv::Mat
  cv::Mat depth_map = computeDepth();

  // store the depth image
  depth_img_ = convertToDepthImg(depth_map);

  // convert to depth cloud
  cv::Mat resized_img;
  cv::resize(input_img, resized_img, cv::Size(resize_w_, resize_h_));
  createPointCloudFromDepth(depth_map, resized_img);
}

cv::Mat MonoDepthEstimation::computeDepth()
{
  cv::Mat result_mat(resize_h_, resize_w_, CV_32F, result_.data());
  cv::Mat scaled_disp = min_disp_ + (max_disp_ - min_disp_) * result_mat;
  cv::Mat depth = 1.0f / scaled_disp;
  depth = STEREO_SCALE_FACTOR * depth;

  return depth;
}

cv::Mat MonoDepthEstimation::convertToDepthImg(const cv::Mat &depth_map)
{
  cv::Mat depth_vis;
  depth_map.convertTo(depth_vis, CV_8UC1, 255.0 / max_depth_);
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  cv::Mat depth_clahe;
  clahe->apply(depth_vis, depth_clahe);
  cv::Mat depth_colormap;
  cv::applyColorMap(depth_clahe, depth_colormap, cv::COLORMAP_JET);

  return depth_colormap;
}

void MonoDepthEstimation::initializeDepthCloud()
{
  // Fill the pcd infomation based on the image
  depth_cloud_.height = resize_h_;
  depth_cloud_.width = resize_w_;
  depth_cloud_.is_bigendian = false;
  depth_cloud_.is_dense = false;

  // Define fields
  if(use_rgb_)
  {
    sensor_msgs::msg::PointField field_x;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;

    sensor_msgs::msg::PointField field_y;
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;

    sensor_msgs::msg::PointField field_z;
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;

    sensor_msgs::msg::PointField field_rgb;
    field_rgb.name = "rgb";
    field_rgb.offset = 12;
    field_rgb.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_rgb.count = 1;

    depth_cloud_.fields.push_back(field_x);
    depth_cloud_.fields.push_back(field_y);
    depth_cloud_.fields.push_back(field_z);
    depth_cloud_.fields.push_back(field_rgb);
    depth_cloud_.point_step = 16;
  }
  else
  {
    sensor_msgs::msg::PointField field_x;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;

    sensor_msgs::msg::PointField field_y;
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;

    sensor_msgs::msg::PointField field_z;
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;

    depth_cloud_.fields.push_back(field_x);
    depth_cloud_.fields.push_back(field_y);
    depth_cloud_.fields.push_back(field_z);
    depth_cloud_.point_step = 12;
  }

  depth_cloud_.row_step = depth_cloud_.point_step * depth_cloud_.width;
  depth_cloud_.data.resize(depth_cloud_.row_step * depth_cloud_.height);
}

void MonoDepthEstimation::createPointCloudFromDepth(const cv::Mat &depth,
                                                    const cv::Mat &rgb)
{
  sensor_msgs::PointCloud2Iterator<float> iter_x(depth_cloud_, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(depth_cloud_, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(depth_cloud_, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_rgb(depth_cloud_, "rgb");

  for(int v = 0; v < depth.rows; ++v)
  {
    for(int u = 0; u < depth.cols; ++u, ++iter_x, ++iter_y, ++iter_z)
    {
      float z;

      z = depth.at<float>(v, u);

      *iter_x = (u - scaled_cx_) * z / scaled_fx_;
      *iter_y = (v - scaled_cy_) * z / scaled_fy_;
      *iter_z = z;

      if(use_rgb_)
      {
        const cv::Vec3b &color = rgb.at<cv::Vec3b>(v, u);
        uint32_t rgb_packed = (color[2] << 16) | (color[1] << 8) | (color[0]);
        float rgb_float;
        std::memcpy(&rgb_float, &rgb_packed, sizeof(float));
        *iter_rgb = rgb_float;
        ++iter_rgb;
      }
    }
  }
}
