from math import frexp
from traceback import print_tb
from torch import imag
from yolov5 import YOLOv5
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
from yolov5_ros2.cv_tool import px2xy
import os
 
# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')
 
# Create a ROS 2 Node class YoloV5Ros2.
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
 
        # Declare ROS parameters.
        self.declare_parameter("device", "cpu", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0"))
 
        self.declare_parameter("model", "best_8_1_2", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))
 
        self.declare_parameter("image_topic", '/D435i/aligned_depth_to_color/image_raw', ParameterDescriptor(
            name="image_topic", description="Image topic, default: /image_raw"))
        
        self.declare_parameter("camera_info_topic", "/camera/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="Camera information topic, default: /camera/camera_info"))
 
        # Read parameters from the camera_info topic if available, otherwise, use the file-defined parameters.
        self.declare_parameter("camera_info_file", f"{package_share_directory}/config/camera_info.yaml", ParameterDescriptor(
            name="camera_info", description=f"Camera information file path, default: {package_share_directory}/config/camera_info.yaml"))
 
        # Default to displaying detection results.
        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))
 
        # Default to publishing detection result images.
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))
 
        # 1. Load the model.
        model_path = package_share_directory + "/config/" + self.get_parameter('model').value + ".pt"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)
 
        # 2. Create publishers.
        self.yolo_result_pub = self.create_publisher(
            Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
 
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)
 
        # 3. Create an image subscriber (subscribe to depth information for 3D cameras, load camera info for 2D cameras).
        # 首先，从ROS 2参数服务器中获取图像话题的名称和相机信息话题的名称。
        # 然后，使用这些话题名称分别创建图像订阅器和相机信息订阅器。
        # 当接收到图像消息时，调用self.image_callback方法处理图像消息。
        # 当接收到相机信息消息时，调用self.camera_info_callback方法处理相机信息消息。
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
 
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 1)    # 从相机信息文件中读取相机参数。
 
        # Get camera information.
        with open(self.get_parameter('camera_info_file').value) as f:
            self.camera_info = yaml.full_load(f.read())
            self.get_logger().info(f"default_camera_info: {self.camera_info['k']} \n {self.camera_info['d']}")
 
        # 4. Image format conversion (using cv_bridge).
        self.bridge = CvBridge()    # 创建一个CvBridge实例，用于图像格式转换。
 
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value
 
    def camera_info_callback(self, msg: CameraInfo):        # 相机信息被提取并存储在camera_info字典中。这个字典被用于存储相机的内参、畸变参数等信息
        """
        Get camera parameters through a callback function.
        """
        self.camera_info['k'] = msg.k       # 相机的内参矩阵，通常是一个 3x3 的矩阵，用来描述相机的焦距和光心位置
        self.camera_info['p'] = msg.p       # 投影矩阵，是相机内参矩阵和相机的畸变参数的组合，用于将相机坐标系中的点投影到图像平面上
        self.camera_info['d'] = msg.d       # 相机的畸变参数，用来描述相机的镜头畸变情况，包括径向畸变和切向畸变
        self.camera_info['r'] = msg.r       # 重投影矩阵，用于立体视觉中的相机标定
        self.camera_info['roi'] = msg.roi   # 感兴趣区域，表示图像中感兴趣的区域的位置和大小
 
        self.camera_info_sub.destroy()
 
    def image_callback(self, msg: Image):
        # 5. Detect and publish results.
        image = self.bridge.imgmsg_to_cv2(msg)      # 将 ROS 消息转换为 OpenCV 格式的图像
        detect_result = self.yolov5.predict(image)  # 使用 YOLOv5 模型对图像进行目标检测，得到检测结果 detect_result
        self.get_logger().info(str(detect_result))
 
        self.result_msg.detections.clear()      # 清空了 self.result_msg 对象中的检测结果，以确保每次处理新的图像时，都能够填充最新的检测结果。
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()
 
        # Parse the results. 
        predictions = detect_result.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
 
        for index in range(len(categories)):
            name = detect_result.names[int(categories[index])]
            detection2d = Detection2D()
            x1, y1, x2, y2 = boxes[index]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1+x2)/2.0
            center_y = (y1+y2)/2.0
 
            if ros_distribution=='galactic':
                detection2d.bbox.center.x = center_x
                detection2d.bbox.center.y = center_y
            else:
                detection2d.bbox.center.x = center_x
                detection2d.bbox.center.y = center_y
 
            detection2d.bbox.size_x = float(x2-x1)
            detection2d.bbox.size_y = float(y2-y1)
 
            obj_pose = ObjectHypothesisWithPose()
            obj_pose.id = name
            obj_pose.score = float(scores[index])
 
            # px2xy
            # px2xy 是一个函数，用于将像素坐标转换为世界坐标。在这里，将目标在图像中的中心像素
            # 坐标 (center_x, center_y) 作为参数传递给 px2xy 函数，同时传入相机的
            # 内参 self.camera_info["k"] 和畸变参数 self.camera_info["d"]。
            # world_x 和 world_y 分别是转换后的目标在相机坐标系中的世界坐标。
            world_x, world_y = px2xy(       
                [center_x, center_y], self.camera_info["k"], self.camera_info["d"], 1)
            obj_pose.pose.pose.position.x = world_x     # 将转换后的世界坐标赋值给目标在 Detection2DArray 消息中的 results 字段中的 pose
            obj_pose.pose.pose.position.y = world_y
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)      # 将结果填充到 Detection2DArray 消息中，包括物体类别、边界框位置、置信度以及物体在相机坐标系中的位置
 
            # Draw results.
            if self.show_result or self.pub_result_img:         #  绘制检测结果并显示
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{name}({world_x:.2f},{world_y:.2f})", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.waitKey(1)
 
        # Display results if needed.
        if self.show_result:
            cv2.imshow('result', image)
            cv2.waitKey(1)
 
        # Publish result images if needed.
        if self.pub_result_img:         # 发布检测结果图像
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            result_img_msg.header = msg.header
            self.result_img_pub.publish(result_img_msg)
 
        if len(categories) > 0:     # 如果检测到物体，就发布 Detection2DArray 消息，其中包含了所有检测到的物体信息
            self.yolo_result_pub.publish(self.result_msg)
 
def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()
 
if __name__ == "__main__":
    main()