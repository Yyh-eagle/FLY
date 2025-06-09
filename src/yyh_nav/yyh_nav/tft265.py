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
from GUI import DebugGUI
import time


class SubscriberNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.l0_vector = np.array([-0.21,0,0])
        self.rot_matrix = None  # 缓存旋转矩阵        
        self.serial = SerialPort()#通信类初始化

        self.pub = self.create_publisher(STM32, "/stm_info", 10)#发布stm32的状态机控制信息给

        self.sub_t265 = self.create_subscription(TFMessage, "/tf", self.T2_listener_callback, 10) #订阅T265的消息
        self.sub_d435 = self.create_subscription(ObjectPosition, "/d435_object_position", self.listener_callback_d435, 10)#订阅D435i发布的目标位置
        self.sub_usb  = self.create_subscription(ObjectPosition,'/usb_object_position',self.listener_callback_usb,10)#订阅usb相机发布的目标位置
        self.tim = self.create_timer(0.1, self.timer_callback) 
        self.tim_serial = self.create_timer(0.01, self.timer_serial_callback)
        # self.gui = DebugGUI(self)  # 创建调试界面
        # self.gui.mainloop()         # 启动GUI主循环

    def timer_serial_callback(self):#100帧发送串口数据
        self.serial.Send_message()
        #self.get_logger().info(f"多及角度: {self.serial.d435_yaw_float}")
        # 分类打印设备状态信息
        #self.get_logger().info(f"T265状态: {'工作' if self.serial.t_flag_u else '未工作'}")
        #self.get_logger().info(f"T265坐标(cm): X={self.serial.T265_x_f:.1f}, Y={self.serial.T265_y_f:.1f}, Z={self.serial.T265_z_f:.1f}")

        #self.get_logger().info(f"D435i状态: {'检测到目标' if self.serial.d_flag_u else '未检测到目标'}, 目标类型: {self.serial.d435_aim_i}")
        if self.serial.d_flag_u:
            self.get_logger().info(f"D435i目标坐标(cm): X={self.serial.d435_x_f:.1f}, Y={self.serial.d435_y_f:.1f}, Z={self.serial.d435_z_f:.1f}")
        #self.get_logger().info(f"USB相机状态: {'检测到目标' if self.serial.c_flag_u else '未检测到目标'}, 目标类型: {self.serial.c_aim_i}")
        # 打印坐标信息（单位cm，保留1位小数）
        # if self.serial.c_flag_u:
        #     self.get_logger().info(f"USB相机目标坐标(cm): X={self.serial.c_x_f:.1f}, Y={self.serial.c_y_f:.1f}")
   

    def timer_callback(self):#10帧，为了
        self.serial.receive()#定时器接受数据
        #将接受到的变量赋值在msg中
        msg = STM32()#成功实现通信
        msg.ifarrive = self.serial.ifArrive_int
        msg.id = self.serial.task_id_int
        msg.state = self.serial.task_state_int
        msg.yaw = self.serial.d435_yaw_float
        msg.z = self.serial.T265_z_f#todo 检验这个消息格式对不对
        self.pub.publish(msg)#发布控制消息

    #D435i回调函数
    def listener_callback_d435(self, msg): 

        self.d435_x,self.d435_y,self.d435_z=msg.x/10,msg.y/10,msg.z/10
        #self.get_logger().info(str(msg.f))
        self.serial.d_flag_u = msg.f#是否有目标
        self.serial.d435_aim_i =msg.kind#目标类型
        self.Q2_eulur2_rotation()#四元数转欧拉角和旋转矩阵
        self.get_world_point()#经过世界坐标转换
        
    def get_world_point(self):
     
        yaw = self.euler[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        x_w = self.d435_x*cos_yaw - self.d435_y*sin_yaw + self.t2x*100
        y_w = self.d435_x*sin_yaw + self.d435_y*cos_yaw + self.t2y*100
        z_w = self.d435_z+self.t2z*100
        #self.get_logger().info("最终得到的坐标:[%d, %d, %d]cm" % (x_w, y_w,z_w))
        self.serial.d435_x_f = -y_w
        self.serial.d435_y_f = x_w
        self.serial.d435_z_f = z_w
    #USB相机回调函数    
    def listener_callback_usb(self,msg):

        
        self.serial.c_flag_u = msg.f#小相机是否检测到目标
        self.serial.c_aim_i = msg.kind#小相机检测目标类别
        x = msg.x/10#小相机的cm目标
        y = msg.y/10
        self.process_USB(x,y)#处理USB相机数据

    
    def process_USB(self,x,y):
        theta = self.euler[2]
        self.serial.c_x_f = (x*np.cos(theta) - y*np.sin(theta))*1.5
        self.serial.c_y_f = (x*np.sin(theta) + y*np.cos(theta))*1.5
        #self.get_logger().info("小相机坐标(cm): X={:.1f}, Y={:.1f}".format(self.serial.c_x_f, self.serial.c_y_f))

          
    #T265的回调函数
    def T2_listener_callback(self,msg):                                             # 创建回调函数，执行收到话题消息后对数据的处理
        for transform in msg.transforms:
            
            translation = transform.transform.translation#获取平移向量
            rotation = transform.transform.rotation#获取旋转向量
            
            self.t2x,self.t2y,self.t2z = translation.x,translation.y,translation.z#赋值平移向量
            self.q0,self.q1,self.q2,self.q3  = rotation.w,rotation.x,rotation.y,rotation.z#赋值旋转向量(四元数)

            
            self.Q2_eulur2_rotation()#四元数转欧拉角和旋转矩阵
            #开始进行T265机身坐标的转换
            r_r = np.dot(self.rot_matrix,self.l0_vector)#用旋转矩阵乘中心相对于T265的向量
            r_e = np.array([self.t2x,self.t2y,self.t2z], dtype=np.float32)
            self.T265_real = r_r + r_e - self.l0_vector#质心加上姿态向量减去初始坐标系偏移
                
            #串口输出数据进行赋值
            self.serial.t_flag_u = 1
            self.serial.T265_x_f = -self.T265_real[1]*100
            self.serial.T265_y_f = self.T265_real[0]*100
            self.serial.T265_z_f = self.T265_real[2]*100
    ##################################工具函数#####################################################
    def Q2_eulur2_rotation(self):
        quaternion = [self.q1, self.q2, self.q3, self.q0]
        r = R.from_quat(quaternion)
        
        self.euler=r.as_euler('xyz', degrees=False)
        self.rot_matrix = r.as_matrix()  # 缓存矩阵    


#入口
def main(args=None):                                 # ROS2节点主入口main函数
    rclpy.init(args=args)                            # ROS2 Python接口初始化
    node = SubscriberNode("tft265")  # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                                 # 循环等待ROS2退出
    node.destroy_node()                              # 销毁节点对象
    rclpy.shutdown()                                 # 关闭ROS2 Python接口
