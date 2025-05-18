#include "mono_depth/depth_estimation.hpp"
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>

MonoDepthEstimation::MonoDepthEstimation(int input_h, int input_w, float fx, float fy,
                                         float cx, float cy,
                                         const std::string &depth_weight_file)
{
  // Set depth img size and detection img size
  resize_h_ = input_h;
  resize_w_ = input_w;

  // Set intrinsic params
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;

  // initialize depth cloud
  initializeDepthCloud();

  // Set up TRT
  initializeTRT(depth_weight_file);

  // Create stream
  cudaStreamCreate(&stream_);
}

cv::Mat MonoDepthEstimation::preprocessImage(const cv::Mat &image, int input_width,
                                             int input_height)
{
  cv::Mat resized, float_image;

  // Resize to model input size
  cv::resize(image, resized, cv::Size(input_width, input_height));

  // Convert to float32
  resized.convertTo(float_image, CV_32F, 1.0 / 255.0); // Normalize to [0,1]

  // Convert from HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  cv::Mat chw_image;
  cv::vconcat(channels, chw_image); // Stack channels in CHW order

  return chw_image;
}

std::vector<float> MonoDepthEstimation::imageToTensor(const cv::Mat &mat)
{
  std::vector<float> tensor_data;
  if(mat.isContinuous())
    tensor_data.assign((float *)mat.datastart, (float *)mat.dataend);
  else
  {
    for(int i = 0; i < mat.rows; i++)
      tensor_data.insert(tensor_data.end(), mat.ptr<float>(i),
                         mat.ptr<float>(i) + mat.cols);
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

  spdlog::info("Image resizing and converting to tensor done");

  // Allocate memory in GPU
  void *buffers_[2];
  float *input_host_ = nullptr;
  float *output_host_ = nullptr;

  cudaMallocHost(reinterpret_cast<void **>(&input_host_),
                 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[0], 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_host_),
                 resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[1], resize_h_ * resize_w_ * sizeof(float));
  // Copy to host memory and then to GPU
  std::memcpy(input_host_, input_tensor.data(),
              1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMemcpyAsync(buffers_[0], input_host_, 1 * 3 * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  spdlog::info("Transfering to cudamalloc host and then to buffer done");
  // Set up inference buffers
  context->setInputTensorAddress("input", buffers_[0]);
  context->setOutputTensorAddress("depth", buffers_[1]);
  spdlog::info("setting input and out done");

  // inference
  context->enqueueV3(stream_);
  spdlog::info("inferencing done");

  // Copy the result back
  cudaMemcpyAsync(output_host_, buffers_[1], resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  spdlog::info("copying output to ouptu host done");
  cudaStreamSynchronize(stream_);

  spdlog::info("stream synchronize done");
  int output_size = resize_h_ * resize_w_;
  result_.assign(output_host_, output_host_ + output_size);

  // store the depth image
  depth_img_ = convertToDepthImg();
  depth_map_ = convertToDepthMap();

  // convert to depth cloud
  cv::Mat resized_img;
  cv::resize(input_img, resized_img, cv::Size(resize_w_, resize_h_));
  createPointCloudFromDepth(depth_map_, resized_img);
  cudaFreeHost(input_host_);
  cudaFreeHost(output_host_);
  cudaFree(buffers_[0]);
  cudaFree(buffers_[1]);
}

cv::Mat MonoDepthEstimation::convertToDepthImg()
{
  spdlog::info("Converting Inference to cv::Mat");

  cv::Mat depth_map(resize_h_, resize_w_, CV_32FC1, result_.data());
  spdlog::info("Conversion Successful");
  depth_map *= MAX_DEPTH;
  cv::Mat depth_vis;
  depth_map.convertTo(depth_vis, CV_8UC1, 255.0 / 80.0);
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  cv::Mat depth_clahe;
  clahe->apply(depth_vis, depth_clahe);
  cv::Mat depth_colormap;
  cv::applyColorMap(depth_clahe, depth_colormap, cv::COLORMAP_JET);

  return depth_colormap;
}

cv::Mat MonoDepthEstimation::convertToDepthMap()
{
  cv::Mat depth_map(resize_h_, resize_w_, CV_32FC1, result_.data());
  depth_map *= 80.0f;
  return depth_map;
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
  spdlog::info("DEPTH CLOUD INITIALIZED");
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

      if(depth.type() == CV_16UC1)
      {
        z = depth.at<uint16_t>(v, u) * 0.001f; // mm to meters
      }
      else if(depth.type() == CV_32FC1)
      {
        z = depth.at<float>(v, u);
      }
      else
      {
        z = 0;
      }

      if(z == 0 || std::isnan(z))
      {
        *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
        if(use_rgb_)
          ++iter_rgb;
        continue;
      }

      *iter_x = (u - cx_) * z / fx_;
      *iter_y = (v - cy_) * z / fy_;
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
