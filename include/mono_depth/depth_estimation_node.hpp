#ifndef DEPTH_ESTIMATION_NODE__DEPTH_ESTIMATION_NODE_HPP_
#define DEPTH_ESTIMATION_NODE__DEPTH_ESTIMATION_NODE_HPP_

#include "mono_depth/depth_estimation.hpp"

#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

class DepthEstimationNode : public rclcpp::Node {
public:
  DepthEstimationNode();

private:
  // Params
  std::string image_topic_;
  std::string depth_weight_file_;
  std::string camera_frame_;
  int depth_input_h_, depth_input_w_;
  int cam_height_, cam_width_;
  int fy_, fx_, cy_, cx_;

  // Variables
  cv::Mat init_image_;
  cv_bridge::CvImagePtr init_image_ptr_;
  Eigen::Matrix3d intrinsic_mat_;
  Eigen::Matrix3d K_inv_;
  std::optional<MonoDepthEstimation> monodepth_;

  // Subscriber
  image_transport::Subscriber image_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Publishers
  image_transport::Publisher depth_img_pub_;

  void timerCallback();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
  void publishDepthImage(const image_transport::Publisher &pub);
};

#endif // GRID_VISION_NODE__GRID_VISION_NODE_HPP_
