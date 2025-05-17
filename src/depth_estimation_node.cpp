#include "mono_depth/depth_estimation_node.hpp"

DepthEstimationNode::DepthEstimationNode() : Node("depth_estimation_node") {
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  depth_weight_file_ = declare_parameter<std::string>("depth_weights_file", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  depth_input_h_ = declare_parameter("depth_network_height", 192);
  depth_input_w_ = declare_parameter("depth_network_width", 640);
  cam_height_ = declare_parameter("camera_image_height", 480);
  cam_width_ = declare_parameter("camera_image_width", 640);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0.0);
  cy_ = declare_parameter("cy", 0.0);

  if (image_topic_.empty() || depth_weight_file_.empty()) {
    RCLCPP_ERROR(get_logger(),
                 "Check if topic name or weight file is assigned");
    return;
  }
  // Image Transport for subscribing
  image_sub_ = image_transport::create_subscription(
      this, image_topic_,
      std::bind(&DepthEstimationNode::imageCallback, this,
                std::placeholders::_1),
      "raw");

  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&DepthEstimationNode::timerCallback, this));

  depth_img_pub_ = image_transport::create_publisher(this, "/depth_image");

  // Get weight paths
  std::string share_dir =
      ament_index_cpp::get_package_share_directory("grid_vision");
  std::string depth_weight_path = share_dir + depth_weight_file_;

  // Initialize TensorRT and depthEstimation class
  monodepth_ =
      MonoDepthEstimation(depth_input_h_, depth_input_w_, depth_weight_path);

  // Set Intrinsic Matrix
  // intrinsic_mat_ = object_detection::setIntrinsicMatrix(fx_, fy_, cx_, cy_);
  // Get Intrinsic Matrix Inverse
  // K_inv_ = object_detection::computeKInverse(intrinsic_mat_);
}

void DepthEstimationNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  // Convert ROS2 image message to OpenCV format
  try {
    init_image_ptr_ = cv_bridge::toCvCopy(msg, "rgb8");

    // Check if the ptr is present
    if (!init_image_ptr_) {
      RCLCPP_ERROR(this->get_logger(),
                   "cv_bridge::toCvCopy() returned nullptr!");
      return;
    }

    // Copy the image
    init_image_ = init_image_ptr_->image;
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return;
  }
}

void DepthEstimationNode::timerCallback() {
  // Check if the image and pointcloud exists
  if (init_image_.empty()) {
    RCLCPP_WARN(this->get_logger(), "Image is missing in DepthEstimationNode");
    return;
  }

  RCLCPP_ERROR(this->get_logger(), "RUN THE DEPTH ESTIMATION");
  monodepth_->runInference(init_image_);
  RCLCPP_ERROR(this->get_logger(), "DEPTH ESTIMATION DONE");

  publishDepthImage(depth_img_pub_);
  RCLCPP_ERROR(this->get_logger(), "LOOP COMPLETE");
}

void DepthEstimationNode::publishDepthImage(
    const image_transport::Publisher &pub) {
  cv::Mat bbox_img = monodepth_->depth_img_;
  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(header, "rgb8", bbox_img).toImageMsg();

  // Publish image
  pub.publish(*msg);
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DepthEstimationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
