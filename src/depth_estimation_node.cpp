#include "mono_depth/depth_estimation_node.hpp"
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>

DepthEstimationNode::DepthEstimationNode() : Node("depth_estimation_node")
{
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  depth_weight_file_ = declare_parameter<std::string>("depth_weights_file", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  depth_input_h_ = declare_parameter("depth_network_height", 192);
  depth_input_w_ = declare_parameter("depth_network_width", 640);
  cam_height_ = declare_parameter("camera_image_height", 480);
  cam_width_ = declare_parameter("camera_image_width", 640);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0.0);
  cy_ = declare_parameter("cy", 0.0);

  if(image_topic_.empty() || depth_weight_file_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Check if topic name or weight file is assigned");
    return;
  }
  // Image Transport for subscribing
  image_sub_ = image_transport::create_subscription(
    this, image_topic_,
    std::bind(&DepthEstimationNode::imageCallback, this, std::placeholders::_1), "raw");

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&DepthEstimationNode::timerCallback, this));

  depth_img_pub_ = image_transport::create_publisher(this, "/depth_image");
  depth_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/depth_cloud", 10);

  // Get weight paths
  std::string share_dir = ament_index_cpp::get_package_share_directory("mono_depth");
  std::string depth_weight_path = share_dir + depth_weight_file_;

  // Initialize TensorRT and depthEstimation class
  monodepth_ = std::make_unique<MonoDepthEstimation>(depth_input_h_, depth_input_w_, fx_,
                                                     fy_, cx_, cy_, depth_weight_path);
  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
}

DepthEstimationNode::~DepthEstimationNode()
{
  timer_->cancel();
  monodepth_.reset();
}

void DepthEstimationNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  // Convert ROS2 image message to OpenCV format
  try
  {
    init_image_ptr_ = cv_bridge::toCvCopy(msg, "rgb8");

    // Check if the ptr is present
    if(!init_image_ptr_)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge::toCvCopy() returned nullptr!");
      return;
    }

    // Copy the image
    init_image_ = init_image_ptr_->image;
  }
  catch(cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return;
  }
}

void DepthEstimationNode::timerCallback()
{
  // Check if the image and pointcloud exists
  if(init_image_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "Image is missing in DepthEstimationNode");
    return;
  }

  auto start_time = std::chrono::steady_clock::now();
  // Run Monocular depth estimation
  monodepth_->runInference(init_image_);

  auto end_time = std::chrono::steady_clock::now();
  auto duration_ms
    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  RCLCPP_INFO(this->get_logger(), "Inference took %ld ms", duration_ms);

  // Publsuh depth image and depth cloud
  publishDepthImage(depth_img_pub_);
  publishDepthCloud(depth_cloud_pub_);
}

void DepthEstimationNode::publishDepthImage(const image_transport::Publisher &pub)
{
  cv::Mat bbox_img = monodepth_->depth_img_;
  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg
    = cv_bridge::CvImage(header, "rgb8", bbox_img).toImageMsg();

  // Publish image
  pub.publish(*msg);
}

void DepthEstimationNode::publishDepthCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub)
{
  // Fill the header
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  header.frame_id = camera_frame_;

  // Set header field
  monodepth_->depth_cloud_.header = header;

  // Transform cloud to base link
  sensor_msgs::msg::PointCloud2 base_cloud
    = transformPointCloud(monodepth_->depth_cloud_, camera_frame_, base_frame_);

  // Publish depth cloud
  pub->publish(base_cloud);
}

sensor_msgs::msg::PointCloud2
DepthEstimationNode::transformPointCloud(const sensor_msgs::msg::PointCloud2 &input_cloud,
                                         const std::string &input_frame,
                                         const std::string &target_frame)
{
  sensor_msgs::msg::PointCloud2 transformed_cloud;
  // Lookup transform from LiDAR to Camera frame
  geometry_msgs::msg::TransformStamped transform_stamped;
  try
  {
    transform_stamped
      = tf_buffer_->lookupTransform(target_frame, input_frame, tf2::TimePointZero);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_ERROR(get_logger(), "Could not transform %s to %s: %s", input_frame.c_str(),
                 target_frame.c_str(), ex.what());
    return transformed_cloud;
  }

  // Apply transformation to point cloud
  pcl_ros::transformPointCloud(target_frame, transform_stamped, input_cloud,
                               transformed_cloud);

  return transformed_cloud;
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DepthEstimationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
