import cv2                             
import numpy as np                      

import pyrealsense2 as rs2#python内置的d435i库
from learning_interface.msg import ObjectPosition

class aim_d4():
    def __init__(self,param):
        self.x = None
        self.y = None
        self.z = None
        self.f = None
        self.kind = None
    
    def update(self,x,y,z,f,kind):
        self.x,self.y,self.z,self.f,self.kind=x,y,z,f,kind


#所有图像模块不需要深度图，只需要彩色图的位置
def locate_d4(param,result):
    """
    输入：彩色图的中心点坐标,舵机偏角
    返回: T265镜头坐标系下的目标位置,直接返回一个数据结构
    """
    angle = param.D435i_yaw#舵机转角

    d_y, d_x = param.depth_d435i.shape[0:2]
    c_y, c_x = param.color_d435i.shape[0:2]

    rate=d_x/c_x#深度图比彩色图的比率
     
    kind_order=result[7]
    real_x_int,real_y_int,real_z_int=GetD435iObject(result,rate,param)#得到目标的T265坐标
    if real_x_int==None:
        return None
    aim = aim_d4(param)
    aim.update(real_x_int,real_y_int,real_z_int,1,kind_order)
    return aim
    #todo还要写出独立测试程序

def GetD435iObject(object,rate,param):
    """
    输入：彩色图的中心点坐标,比率，参数集合
    返回: T265镜头坐标系下的目标位置,直接返回一个数据结构
    """
    
    #yaw = param.D435i_yaw#舵机转角
    yaw = 45/57.3
    center_x_int=int(rate*object[0])#中心点坐标
    center_y_int=int(rate*object[1])
    real_z=param.depth_d435i[center_y_int,center_x_int]#目标深度值获取
    camera_coordinate = rs2.rs2_deproject_pixel_to_point(param.intrinsics, [int(object[0]),int(object[1])], real_z)
    (real_x,real_y,real_z) = camera_coordinate[0:3]#直接小孔成像模型计算了
    if abs(real_x)>1e-3 and abs(real_y)>1e-3 and abs(real_z)>1e-3:

        #先假设都是45度#todo少一个根据角度计算到T265坐标系的转换函数#todoT265坐标系和相机坐标系的方向
        X0 = np.array([real_y, real_z])
        R = np.array([[np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
        X_trans = np.dot(R, X0)
        #param.logger.info(str(X0))
        real_x_int,real_y_int,real_z_int = -int(X_trans[1])-337,int(real_x),-int(X_trans[0])#全是mm级别
        cv2.putText(param.color_d435i,"("+str(real_x_int)+","+str(real_y_int)+","+str(real_z_int)+")",(center_x_int,center_y_int ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    

        return real_x_int,real_y_int,real_z_int
    else:return None,None,None