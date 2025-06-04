#ros库
import rclpy                           
from rclpy.node import Node             

#ros消息库
from sensor_msgs.msg import Image       
from cv_bridge import CvBridge         
from rcl_interfaces.msg import ParameterDescriptor
from learning_interface.msg import ObjectPosition,MyState,STM32  
from message_filters import ApproximateTimeSynchronizer, Subscriber

#计算库
import cv2                             
import numpy as np                      
import math

#系统库
import sys
sys.path.append('/home/yyh/ros2_ws/src/yyh_object/yyh_object/')
sys.path.append('/home/yyh/yolov10/yolov10-main')
import os
import datetime

#自编写库
from yolo_function import *
from circle import *
from ultralytics import YOLOv10
from Code2D import Code2D
"""
所有的功能函数采用同样的返回值格式，[center_x,center_y,x,y,w,confidence,kind]
该格式传入locate_d4函数，返回aim
再将aim传入到Aim2Object函数，返回object
所有传入的参数都是param
"""
#订阅节点类
class ImageSubscriber(Node):
    def __init__(self, name):
        
        super().__init__(name) 
        #计数器   
        self.yolo_cnt = 0
        self.no_aim_cnt_usb = 0
        self.no_aim_cnt_d4 = 0
        self.cv_bridge = CvBridge()
        
        self.param = Param(self.get_logger())#传递参数初始化
        #d435i镜头的初始化以及消息格式
        self.colord435i = None#d435i图像
        self.depthd435i = None#d435i深度图
        self.aim_d4 = None
        self.d4_obj = Aim2Object()
        #usb镜头的初始化以及消息格式
        self.usb = cv2.VideoCapture('/dev/usb')
        if not self.usb.isOpened():
            print("无法打开摄像头")
            exit()
        self.aim_usb =None
        self.usb_obj = Aim2Object()
        #yolo
        self.yolov10 = YOLOv10("/home/yyh/ros2_ws/src/yyh_object/yyh_object/best.pt")


        self.sub_stm = self.create_subscription(STM32, "/stm_info", self.listener_callback_stm, 10)
        color_sub = Subscriber(self, Image, '/D435i/color/image_raw')
        depth_sub = Subscriber(self, Image, '/D435i/aligned_depth_to_color/image_raw')
        self.ts = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.unified_callback)
        self.pub_d435 = self.create_publisher(ObjectPosition, "d435_object_position", 10)             
        self.pub_usb = self.create_publisher(ObjectPosition, "usb_object_position", 10) 

    def unified_callback(self, color_msg, depth_msg):
        """同步处理2种数据"""
        self.param.d435i_color = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        self.param.d435i_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        _, self.param.usb = self.usb.read()
       # self.get_logger().info(f"{self.param.usb}")
        #任务规划核心函数
        self.task_plan(self.param)
        #self._logger.info("in")
        cv2.imshow("D435i", self.param.d435i_color)
        cv2.imshow("USB",self.param.usb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
    #-----------------------------------------------------------------------------#
    #         任务规划核心函数  
    #-----------------------------------------------------------------------------#
        
    def task_plan(self,param):
        #------------------------------------------------------------#
        #  ts:0 tid:0  ============>yolo推理
        #------------------------------------------------------------#
     
        if self.param.task_state == 0:
            if self.param.task_id == 0 :
                if self.yolo_cnt%4==0:#每四帧执行一次yolo推理
                    self.aim_d4 = None
                    self.aim_usb = None
                    if param.ifarrive==1:
                        
                        self.aim_d4 = yolo_d4(param,self.yolov10)
                        #self.get_logger().info(f"{aim_d4}")
                    else:#飞到目标位置后用下视镜头识别
                        self.aim_usb = yolo_usb(param,self.yolov10)
                self.yolo_cnt+=1#用于控制yolo的计数
            #------------------------------------------------------------#
            #  ts:1  ============>二维码识别
            #------------------------------------------------------------#    
            elif self.param.task_id == 1:
                aim = Code2D(param)
        #------------------------------------------------------------#
        #  发布aim消息
        #------------------------------------------------------------#
        """d435i"""
        
        if self.aim_d4 is not None:
            
            object_d4 = self.d4_obj.Update(self.aim_d4)
            
            self.pub_d435.publish(object_d4)
            self.no_aim_cnt_d4 = 0
        else:
            
            self.no_aim_cnt_d4 += 1
            object_d4 = self.d4_obj.object
            if self.no_aim_cnt_d4 >=10:
                object_d4  =self.d4_obj.Clear()
                
            self.pub_d435.publish(object_d4)
        if self.aim_d4 is not None:
            cv2.putText(self.param.d435i_color, f"x :{self.aim_d4.x/10}, y: {self.aim_d4.y/10}, z: {self.aim_d4.z/10}, kind:{self.aim_d4.kind}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        """usb"""
        if self.aim_usb is not None:
            
            object_usb = self.usb_obj.Update(self.aim_usb)
            self.pub_usb.publish(object_usb)
            self.no_aim_cnt_usb = 0
        else:
            
            self.no_aim_cnt_usb += 1
            object_usb = self.usb_obj.object
            if self.no_aim_cnt_usb >=10:
                object_usb  =self.d4_obj.Clear()        
                self.pub_usb.publish(object_usb)
        if self.aim_usb is not None:
            cv2.putText(self.param.usb, f" x:{self.aim_usb.x/10}, y: {self.aim_usb.y/10}, z: {self.aim_usb.z/10}, kind: {self.aim_usb.kind}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
    def __del__(self):
        cv2.destroyAllWindows() 
                
       
    def listener_callback_stm(self,msg):#主任务在这里写，可以保证周期话运行
    #每次收到stm32的信息，都赋值给全局变量
        self.ifarrive = msg.ifarrive
        self.task_id = msg.id
        self.task_state = msg.state
        self.D435i_yaw = msg.yaw
        self.z = msg.z
        self.param.update_param(self.ifarrive,self.task_state,self.task_id,self.D435i_yaw,self.z)

#-----------------------------------------------------------------------------#
#       发布镜头消息类
#-----------------------------------------------------------------------------#
class Aim2Object():

    def __init__(self):
        self.object = ObjectPosition()

    def Update(self,aim):
        self.object.x = aim.x
        self.object.y = aim.y
        self.object.z = aim.z
        self.object.f = aim.f
        self.object.kind = aim.kind
        return self.object
    
    def Clear(self):
        self.object.x = 0
        self.object.y = 0
        self.object.z = 0
        self.object.f = 0
        self.object.kind = 0
        return self.object


#-----------------------------------------------------------------------------#
#       参数类
#-----------------------------------------------------------------------------#
class Param():
    def __init__(self,logger):
        self.ifarrive = 0
        self.task_state = 0
        self.task_id = 0
        self.D435i_yaw = 0.0
        self.d435i_color = None
        self.d435i_depth = None
        self.frame = None
        self.logger = logger
        self.usb =None


    def update_param(self,ifarrive,task_state,task_id,D435i_yaw,z):
        self.ifarrive = ifarrive
        self.task_state = task_state
        self.task_id = task_id
        self.D435i_yaw = D435i_yaw
        self.z = z



def main(args=None):                            # ROS2节点主入口main函数()
    rclpy.init(args=args)                       # ROS2 Python接口初始化
    node = ImageSubscriber("image")  # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                            # 循环等待ROS2退出
    node.destroy_node()                         # 销毁节点对象
    rclpy.shutdown()                            # 关闭ROS2 Python接口