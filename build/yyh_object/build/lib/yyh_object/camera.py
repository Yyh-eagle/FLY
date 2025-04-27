import rclpy                            # ROS2 Python接口库
from rclpy.node import Node             # ROS2 节点类
from sensor_msgs.msg import Image       # 图像消息类型
from cv_bridge import CvBridge          # ROS与OpenCV图像转换类
from rcl_interfaces.msg import ParameterDescriptor
import message_filters
from learning_interface.msg import ObjectPosition,MyState  # 自定义的目标位置消息


import cv2                              # Opencv图像处理库
import numpy as np                      # Python数值计算库
import math
import time


import pyzbar.pyzbar as pyzbar# 2dcode
from ament_index_python.packages import get_package_share_directory
from yolov5 import YOLOv5
lower_red = np.array([0, 90, 128])      # 红色的HSV阈值下限
upper_red = np.array([180, 255, 255])   # 红色的HSV阈值上限

package_share_directory = get_package_share_directory('yolov5_ros2')

#创建一个订阅者节点

class ImageSubscriber(Node):
    def __init__(self, name):
        
        super().__init__(name)    
        
        self.pub_usb = self.create_publisher(
            ObjectPosition, "usb_object_position", 10)              # 创建发布者对象（消息类型、话题名、队列长度）
        
        #########################################yolo#########################################
        #No cuda so cpu
        self.declare_parameter("device", "cpu", ParameterDescriptor(
            name="device", description="calculate_device default:cpu optional:cuda:0"))

        self.declare_parameter("model", f"{package_share_directory}/config/last.pt", ParameterDescriptor(
            name="model", description=f"default: {package_share_directory}/config/best_landing1.pt"))

        self.declare_parameter("image_topic", "/image_raw", ParameterDescriptor(
            name="image_topic", description=f"default: /image_raw"))

        self.declare_parameter("camera_info_topic", "/camera/camera_info", ParameterDescriptor(
            name="camera_info_topic", description=f"default: /camera/camera_info")) 

        # 默认从camera_info中读取参数,如果可以从话题接收到参数则覆盖文件中的参数
        self.declare_parameter("camera_info_file", f"{package_share_directory}/config/camera_info.yaml", ParameterDescriptor(
            name="camera_info", description=f"{package_share_directory}/config/camera_info.yaml"))
        
        
        # 1.load model
        model = self.get_parameter('model').value
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model, device=device)
        
       
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.usb = None
        self.cap = cv2.VideoCapture(0)
        # 创建定时器，每隔0.1秒调用一次 image_process
        self.timer = self.create_timer(0.01, self.image_process)

        
        
    

    
    def yolo_recog(self,img,threshold):#yolo for d435i
        detect_result = self.yolov5.predict(img)#predict
        predictions = detect_result.pred[0]#result
        boxes = predictions[:, :4]  # x1, y1, x2, y2  = predictions[:,0],predictions[:,1],predictions[:,2],predictions[:,3]
        scores = predictions[:, 4]#score = predictions[:,4]
        categories = predictions[:, 5]#cate = predictions[:,5]
        result=[]
        kinds=['landing_cross','one','two','three','four','A']
        for index in range(len(categories)):
            name = detect_result.names[int(categories[index])]
            for j in range(len(kinds)):
                if kinds[j] ==name:

                    name_order=j#find the same kind the same with category
                    break
                
            x1, y1, x2, y2 = boxes[index]#find th center of the box
            center_x=(x1+x2)/2
            center_y=(y1+y2)/2
            x1_int=int(x1)
            x2_int=int(x2)
            y1_int=int(y1)
            y2_int=int(y2)
            center_x_int=int(center_x)
            center_y_int=int(center_y)

            probability= detect_result.pred[0][index][4]
            if probability>threshold:
                pack=(center_x_int,center_y_int,x1_int,y1_int,x2_int,y2_int,probability,name_order)
                result.append(pack)#turple
    
        return result
    
    def locate_yolo_usb(self,color_image):
        #self.get_logger().info('d_x=%d,d_y=%d  c_x=%d,c_y=%d,c_z=%d' % (d_x,d_y,c_x,c_y,c_z))   
        flag_find=0
        target_x=0
        target_y=0
        gray_img=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        results=self.yolo_recog(gray_img,0.80)
        kind_order=-1
        for object in results:
            x1=object[2]
            y1=object[3]
            x2=object[4]
            y2=object[5]
            probability=object[6]
            kind_order=object[7]
            center_x_int=int(object[0])
            center_y_int=int(object[1])
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)   
            cv2.putText(color_image,str(kind_order),(x1,y1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            height, width = color_image.shape[0:2]
            half_height=int(height//2)
            half_wid=int(width//2)
            target_x=(half_wid-center_x_int)
            target_y=(center_y_int-half_height)
            flag_find=1
            break
        position = ObjectPosition()
        if kind_order==0:
            position.x[0], position.y[0],position.z[0],position.f,position.kind = target_y,-target_x,0,flag_find,13
            print(position.x[0], position.y[0],position.z[0])
        self.pub_usb.publish(position)
        cv2.imshow("color",color_image)                          # 使用OpenCV显示处理后的图像效果
        cv2.waitKey(1)
    
    def decodeDisplay(self):
        img=cv2.cvtColor(self.usb,cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(self.usb)
        rects_list = []
        QR_info = []
        results=[]
        # 这里循环，因为画面中可能有多个二维码
        for barcode in barcodes:
            # 提取条形码的边界框的位置
            # 画出图像中条形码的边界框
            (x, y, w, h) = barcode.rect
            cx=x+w//2
            cy=y+h//2
            rects_list.append((x, y, w, h))
            cv2.rectangle(self.usb, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # 条形码数据为字节对象，所以如果我们想在输出图像上画出来，就需要先将它转换成字符串
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            location=[cx,cy]
            # 绘出图像上条形码的数据和条形码类型
            text = str(barcodeData)
            result=[text,location]
            results.append(result)
            cv2.putText(self.usb, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 125), 2)
            # 向终端打印条形码数据和条形码类型
            # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        return results

    #颜色识别
    def color_detect(self,image):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)      # 图像从BGR颜色模型转换为HSV模型
        mask_red = cv2.inRange(hsv_img, lower_red, upper_red) # 图像二值化
        contours, hierarchy = cv2.findContours(
            mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   # 图像中轮廓检测
 
        for cnt in contours:                                  # 去除一些轮廓面积太小的噪声
            if cnt.shape[0] < 150:
                continue
 
            (x, y, w, h) = cv2.boundingRect(cnt)              # 得到苹果所在轮廓的左上角xy像素坐标及轮廓范围的宽和高
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)# 将苹果的轮廓勾勒出来
            cv2.circle(image, (int(x+w/2), int(y+h/2)), 5,
                       (0, 255, 0), -1)                       # 将苹果的图像中心点画出来
 
        cv2.imshow("object", image)                           # 使用OpenCV显示处理后的图像效果
        cv2.waitKey(10)
        
    #circle detect
    def houf_circle(self,image):
        
        #time =0.02s
        minRadius=25
        maxRadius=600
        circles = []
        e1=cv2.getTickCount()
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_image,(5,5),0)
        ret,thresh = cv2.threshold(gray_image,125,255,cv2.THRESH_BINARY)
        
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, (-1, -1))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, (-1, -1))

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) <5:
                continue
                
            area = cv2.contourArea(cnt)
            if area < (minRadius**2) * math.pi or area > (maxRadius**2) * math.pi:
                continue
            
            arc_length = cv2.arcLength(cnt, True)
            radius = arc_length / (2 * math.pi)
            
            if not (minRadius < radius and radius < maxRadius):
                continue
            
            ellipse = cv2.fitEllipse(cnt)
            ratio = float(ellipse[1][0]) / float(ellipse[1][1])
            
            if ratio > 0.8 and ratio < 1.2:#e
                corner = cv2.approxPolyDP(cnt, 0.02 * arc_length, True)
                cornerNum = len(corner)
                if cornerNum > 4: # 当cornerNum=4时，识别矩形；而cornerNum>4时，识别圆
                    circles.append(ellipse)
                    
            for circle in circles:
                print(circle)
                cv2.circle(image, (int(circle[0][0]), int(circle[0][1])), int(0.25*(circle[1][0]+circle[1][1])), (0, 255, 0), thickness=5)
        
        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        #print(circles)
        cv2.imshow("hourf", image)                           # 使用OpenCV显示处理后的图像效果
        cv2.waitKey(10)  
        #print(time)
                    
    def image_process(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to capture frame")
            return

        # 处理图像，例如每3帧进行一次YOLO检测
        if self.frame_count % 3 == 0:
            self.locate_yolo_usb(frame)

        self.frame_count += 1

    
       
                    
    
        
 
 
def main(args=None):                            # ROS2节点主入口main函数()
    rclpy.init(args=args)                       # ROS2 Python接口初始化
    node = ImageSubscriber("camera")  # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                            # 循环等待ROS2退出
    node.destroy_node()                         # 销毁节点对象
    rclpy.shutdown()                            # 关闭ROS2 Python接口