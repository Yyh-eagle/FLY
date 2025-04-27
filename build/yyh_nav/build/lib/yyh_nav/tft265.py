import rclpy                                       # ROS2 Python接口库
from rclpy.node   import Node                      # ROS2 节点类
from std_msgs.msg import String                    # 字符串消息类型
from tf2_msgs.msg import TFMessage
from learning_interface.msg import ObjectPosition  # 自定义的目标位置消息
from learning_interface.msg import STM32


import numpy as np 
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot 


#引入其他函数
import sys
sys.path.append('/home/yyh/ros2_ws/src/yyh_nav/yyh_nav/')
from Myserial import SerialPort
from data_vision import ImageVisualizer
import time




class SubscriberNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.serial = SerialPort()#通信类初始化
        self.vision =  ImageVisualizer()
        #创建发布者
        self.pub = self.create_publisher(STM32, "/stm_info", 10)#stm32的状态机控制信息
        # 创建订阅者
        self.sub = self.create_subscription(\
            TFMessage, "/tf", self.T2_listener_callback, 10) #订阅T265的消息
        self.sub_d435 = self.create_subscription(\
            ObjectPosition, "/d435_object_position", self.listener_callback_d435, 10)#不订阅图像，只保留目标信息
        self.sub_usb  = self.create_subscription(\
            ObjectPosition,'/usb_object_position',self.listener_callback_usb,10)
        self.tim = self.create_timer(0.01, self.timer_callback) #主循环 10帧
        
        
    def timer_callback(self):
        #self.serial.Timer_zero()#清零所需变量
        self.serial.receive()#定时器接受数据
        #self.get_logger().info(str(self.serial.receive_num))
        #将接受到的变量赋值在msg中
        msg = STM32()#成功实现通信
        msg.ifarrive = self.serial.ifArrive_int
        msg.id = self.serial.task_id_int
        msg.state = self.serial.task_state_int
        msg.yaw = self.serial.d435_yaw_float
        self.pub.publish(msg)
    

        

    #T265回调函数 
    def T2_listener_callback(self,msg):                                             # 创建回调函数，执行收到话题消息后对数据的处理
        
        for transform in msg.transforms:
            #接受数据
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            #数据赋值
            self.r_a =None
            self.t2x,self.t2y,self.t2z = translation.x,translation.y,translation.z
            self.q0,self.q1,self.q2,self.q3  = rotation.w,rotation.x,rotation.y,rotation.z
        
            #旋转矩阵初始化
            self.Q2E_Q2R()
            self.correctT265()
    def Q2E_Q2R(self):#四元数转欧拉角和旋转矩阵
        quaternion = [self.q1, self.q2, self.q3, self.q0]
        r = R.from_quat(quaternion)
        self.euler=r.as_euler('xyz', degrees=True)/57.3#获取欧拉角
        
        self.rot = R.from_quat(quaternion).as_matrix()
        

    #将T265镜头转换为机身中心位置和姿态
    def correctT265(self):
        self.l0_vector = np.array([-0.197,0,0])
        r_r = np.dot(self.rot,self.l0_vector)#用旋转矩阵乘中心相对于T265的向量
        r_e = np.array([self.t2x,self.t2y,self.t2z])
        self.r_a = r_r + r_e - self.l0_vector#质心加上姿态向量减去初始坐标系偏移
        
        #为了串口输出数据
        self.serial.t_flag_u = 1
        self.serial.T265_x_f = -self.r_a[1]*100
        self.serial.T265_y_f = self.r_a[0]*100
        self.serial.T265_z_f = self.r_a[2]*100
        
        self.serial.Send_message()
        #self.get_logger().info(str(self.serial.data_num))
        self.vision.update_variable("T265_x",self.serial.T265_x_f)
        self.vision.update_variable("T265_y",self.serial.T265_y_f)
        self.vision.update_variable("T265_z",self.serial.T265_z_f)
        
    def listener_callback_d435(self, msg): 



        self.d435_x,self.d435_y,self.d435_z=msg.x/10,msg.y/10,msg.z/10
        self.serial.d435_flag_u = msg.f#是否有目标
        self.serial.D435_aim_i =msg.kind#目标是什么类型
        
        self.get_world_point()#d经过世界坐标转换
        
    def get_world_point(self):
     

        yaw = self.euler[2]

        #self.get_logger().info("t265坐标系下的目标点: %d, %d, %d" % (self.t2x*100, self.t2y*100, self.t2z*100))
        x_w = self.d435_x*np.cos(yaw) - self.d435_y*np.sin(yaw) + self.t2x*100
        y_w = self.d435_x*np.sin(yaw) + self.d435_y*np.cos(yaw) + self.t2y*100
        z_w = self.d435_z+self.t2z*100
        #self.get_logger().info("最终得到的坐标:[%d, %d, %d]cm" % (x_w, y_w,z_w))
        self.serial.d435_x_f = x_w
        self.serial.d435_y_f = y_w
        self.serial.d435_z_f = z_w
       # self.vision.update_variable("T265_obj_x",self.d435_x)
       # self.vision.update_variable("T265_obj_y",self.d435_y)
       # self.vision.update_variable("T265_obj_z",self.d435_z)
        #self.vision.update_variable("world_obj_x",self.serial.d435_x_f)
        # self.vision.update_variable("world_obj_y",self.serial.d435_y_f)
        # self.vision.update_variable("world_obj_z",self.serial.d435_z_f)
    def listener_callback_usb(self,msg):
        self.usb_x=msg.x
        self.usb_y=msg.y
        self.flag_u=msg.f
        print("3",msg.x)
        self.get_logger().info('Target Position: "(%d,%d,%d)"' % (self.usb_x[0], self.usb_x[0],self.usb_x[0]))
        

def main(args=None):                                 # ROS2节点主入口main函数
    rclpy.init(args=args)                            # ROS2 Python接口初始化
    node = SubscriberNode("tft265")  # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                                 # 循环等待ROS2退出
    node.destroy_node()                              # 销毁节点对象
    rclpy.shutdown()                                 # 关闭ROS2 Python接口