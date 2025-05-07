import cv2                             
import numpy as np                      
from learning_interface.msg import ObjectPosition


"""
本文件用于形成坐标系转换,将像素坐标系转换为T265的相机坐标
"""
#------------------------------------------------------------#
#  像素坐标系转换为T265坐标后的返回值格式定义
#------------------------------------------------------------#
class Aim():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.f = None
        self.kind = None
    
    def update(self,x,y,z,f,kind):
        self.x,self.y,self.z,self.f,self.kind=x,y,z,f,kind

#------------------------------------------------------------#
#  d435i的坐标系转换为T265坐标系
#------------------------------------------------------------#
def locate_d4(param,result):

    rate=param.d435i_depth.shape[0]/param.d435i_color.shape[0]#获取比率
     
    kind_order=result[7]

    real_x_int,real_y_int,real_z_int=GetD435iObject(result,rate,param)#得到目标相对于T265镜头定义的坐标系下的坐标
    
    aim = Aim()#创建一个通信格式
    aim.update(real_x_int,real_y_int,real_z_int,1,kind_order)

    return aim
    

def GetD435iObject(object,rate,param):
   
    #yaw = param.D435i_yaw#舵机转角#todo需要刘华强给我传数据
    yaw = 45/57.3
    center_x_int=int(rate*object[0])#中心点坐标
    center_y_int=int(rate*object[1])
    real_z=param.d435i_depth[center_y_int,center_x_int]#目标深度值获取
    if real_z <= 0.4 :
        real_z = 0.4#保证不会出现z消失的情况
    camera_coordinate = pixel_to_camera_coordinate(center_x_int,center_y_int,real_z,\
                                        fx =608.034 , fy=607.711, cx=430, cy=251.383)#小孔成像模型计算
    (real_x,real_y,real_z) = camera_coordinate[0:3]#小孔成像模型计算
   
    
    X0 = np.array([real_y, real_z])
    R = np.array([[np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]])
    X_trans = np.dot(R, X0)
    #param.logger.info(str(X0))
    real_x_int,real_y_int,real_z_int = -int(X_trans[1])-337,int(real_x),-int(X_trans[0])#todo需要测量两个镜头之间的距离
    cv2.putText(param.d435i_color,
                "(" + str(real_x_int) + "," + str(real_y_int) + "," + str(real_z_int) + ")",
                (center_x_int, center_y_int),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2)

    return real_x_int,real_y_int,real_z_int
  


#------------------------------------------------------------#
#  对于usb相机
#------------------------------------------------------------#

def locate_usb(param,result):
    """参数含义：real_z:usb相机所在高度"""#todo需要tfT265节点发布高度，并减去飞机厚度

    
    center_x,center_y,kind_order = result[0],result[1],result[7]
    real_x,real_y,real_z = pixel_to_camera_coordinate(center_x,center_y,param.z,\
                            fx =608.034 , fy=607.711, cx=430, cy=251.383)
    
    aim = Aim()#创建一个通信格式
    aim.update(real_x,real_y,real_z,1,kind_order)
    return aim


#------------------------------------------------------------#
#  小孔成像
#------------------------------------------------------------#
#todo需要对D435i的像素进行固定,固定为848,480
def pixel_to_camera_coordinate(px, py,real_z ,fx =608.034 , fy=607.711, cx=430, cy=251.383):#todo需要对相机进行标定
    
    # 小孔成像模型计算
    real_x = (px - cx) * real_z / fx
    real_y = (py - cy) * real_z / fy
    
    return (real_x, real_y, real_z)