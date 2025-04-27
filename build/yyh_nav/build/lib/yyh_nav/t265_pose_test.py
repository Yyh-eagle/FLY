import rclpy                                       # ROS2 Python接口库
from rclpy.node   import Node                      # ROS2 节点类
from std_msgs.msg import String                    # 字符串消息类型
from tf2_msgs.msg import TFMessage
from learning_interface.msg import ObjectPosition  # 自定义的目标位置消息



import numpy as np 
from scipy.spatial.transform import Rotation as R



class SubscriberNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.l0_vector = np.array([-0.197,0,0])
        # 创建订阅者
        self.sub = self.create_subscription(\
            TFMessage, "/tf", self.T2_listener_callback, 10)       # 创建订阅者对象（消息类型、话题名、订阅者回调函数、队列长度
    
        
   
        
    #T265回调函数 
    def T2_listener_callback(self,msg):                                             # 创建回调函数，执行收到话题消息后对数据的处理
        
        for transform in msg.transforms:
            #接受数据
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            #数据赋值
            self.r_a =None
            self.t2x,self.t2y,self.t2z = translation.x,translation.y,translation.z
            #self.get_logger().info('T265 Position: "(%d,%d,%d)"' % ((self.t2x*1000), (self.t2y*1000), (self.t2z*1000)))
            self.q0,self.q1,self.q2,self.q3  = rotation.w,rotation.x,rotation.y,rotation.z
            #旋转矩阵初始化
            self.Q2E_Q2R()
            self.correctT265()
    def Q2E_Q2R(self):#四元数转欧拉角和旋转矩阵
        quaternion = [self.q1, self.q2, self.q3, self.q0]
        r = R.from_quat(quaternion)
        self.euler=r.as_euler('xyz', degrees=True)
        self.rot = R.from_quat(quaternion).as_matrix()
        #self.get_logger().info('T265 Euler Angle: "(%d,%d,%d)"' % (self.euler[0], self.euler[1], self.euler[2]))

    #将T265镜头转换为机身中心位置和姿态
    def correctT265(self):#todo进行飞机的姿态测试单元编写
        
        r_r = np.dot(self.rot,self.l0_vector)#用旋转矩阵乘中心相对于T265的向量
        r_e = np.array([self.t2x,self.t2y,self.t2z])
        #self.get_logger().info('中间结果："(%d,%d,%d)"' % ((r_r+r_e)[0]*1000, (r_r+r_e)[1]*1000, (r_r+r_e)[2]*1000))

        self.r_a = r_r + r_e - self.l0_vector#质心加上姿态向量减去初始坐标系偏移

        self.get_logger().info('Corrected T265 Position: "(%d,%d,%d)"' % (self.r_a[0]*100, self.r_a[1]*100, self.r_a[2]*100))
        #为了串口输出数据
 
      
        

def main(args=None):                                 # ROS2节点主入口main函数
    rclpy.init(args=args)                            # ROS2 Python接口初始化
    node = SubscriberNode("tft265")  # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                                 # 循环等待ROS2退出
    node.destroy_node()                              # 销毁节点对象
    rclpy.shutdown()                                 # 关闭ROS2 Python接口