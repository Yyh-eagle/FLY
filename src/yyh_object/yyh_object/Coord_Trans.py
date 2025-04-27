import cv2                             
import numpy as np                      
from learning_interface.msg import ObjectPosition


"""
本文件用于形成坐标系转换，将像素坐标系转换为T265的相机坐标
"""

class aim_d4():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.f = None
        self.kind = None
    
    def update(self,x,y,z,f,kind):
        self.x,self.y,self.z,self.f,self.kind=x,y,z,f,kind

#思考如何计数为无目标
def locate_d4(param,result):

    rate=param.d435i_depth.shape[0]/param.d435i_color.shape[0]#获取比率
     
    kind_order=result[7]

    real_x_int,real_y_int,real_z_int=GetD435iObject(result,rate,param)#得到目标相对于T265镜头定义的坐标系下的坐标
    
    aim = aim_d4()#创建一个通信格式，专门用于发送d435i的数据
    aim.update(real_x_int,real_y_int,real_z_int,1,kind_order)

    return aim
    

def GetD435iObject(object,rate,param):
   
    #yaw = param.D435i_yaw#舵机转角
    yaw = 45/57.3
    center_x_int=int(rate*object[0])#中心点坐标
    center_y_int=int(rate*object[1])
    real_z=param.d435i_depth[center_y_int,center_x_int]#目标深度值获取
    if real_z <= 0.4 :
        real_z = 0.4#保证不会出现z消失的情况
    camera_coordinate = pixel_to_camera_coordinate(center_x_int,center_y_int,real_z)#小孔成像模型计算
    (real_x,real_y,real_z) = camera_coordinate[0:3]#小孔成像模型计算
   
    
    X0 = np.array([real_y, real_z])
    R = np.array([[np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]])
    X_trans = np.dot(R, X0)
    #param.logger.info(str(X0))
    real_x_int,real_y_int,real_z_int = -int(X_trans[1])-337,int(real_x),-int(X_trans[0])#全是mm级别
    cv2.putText(param.d435i_color,
                "(" + str(real_x_int) + "," + str(real_y_int) + "," + str(real_z_int) + ")",
                (center_x_int, center_y_int),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2)

    return real_x_int,real_y_int,real_z_int
  

def pixel_to_camera_coordinate(px, py,real_z ,fx =608.034 , fy=607.711, cx=430, cy=251.383):
    
  
    if cx is None or cy is None:
        # 注意：这里需要你知道原始图像尺寸，这里假设为1280x720为例
        cx = 1280 / 2
        cy = 720 / 2
    
    # 小孔成像模型计算
    real_x = (px - cx) * real_z / fx
    real_y = (py - cy) * real_z / fy
    
    return (real_x, real_y, real_z)