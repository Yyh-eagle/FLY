import cv2                             
import numpy as np                      


import pyrealsense2 as rs2#python内置的d435i库
from learning_interface.msg import ObjectPosition
from Coord_Trans import locate_d4
YOLO_THRESHOLD = 0.5
"""本文件用于yolo的接口函数"""



#------------------------------------------------------------#
#  yolo和Image文件的接口
#------------------------------------------------------------#
def yolo_d4(param,yolo):
    """d435i的yolo函数"""
    d435i_color = param.d435i_color
    result=yolo_recog_V10(d435i_color,YOLO_THRESHOLD,yolo)

    if result is None:
        return None

    return locate_d4(param,result) 

def yolo_usb(param,yolo):#todo测试usb相机是否可用yolo
    """usb的yolo函数"""
    usb = param.usb
    result=yolo_recog_V10(param,YOLO_THRESHOLD,yolo)

    if result is None:
        return None
        
    return locate_usb(param,result) #todo测试z这一串好不好用

#------------------------------------------------------------#
#  yolo识别并筛选目标
#------------------------------------------------------------# 
def yolo_recog_V10(frame, threshold, yolo):
    class_dict = {0:"landing_cross",1:"landing_square",2:"landing_triangle"}#todo需要根据比赛具体细化
    #-------------------------------------------------------#
    #  用1/4的尺寸进行推理检测
    #---------------------------------------------------  --# 
    original_height, original_width =frame.shape[:2]
    resized_image = cv2.resize(frame, (original_width//4, original_height//4))  # (width, height)
    detect_result = yolo.predict(resized_image,imgsz=original_width//4,half=False,max_det=1,verbose=False)       # 关闭调试输出)  # 用缩放后的图像做预测
    result = detect_result[0]  

    #-------------------------------------------------------#
    #  处理检测结果
    #---------------------------------------------------  --# 
    for box in result.boxes:
       
        name = result.names[int(box.cls)]
        class_id = class_dict.get(name, -1)  # 获取类别ID，默认为-1表示未找到
        # 跳过未识别的类别
        if class_id == -1:
            continue 

        conf = float(box.conf)
        #if name == "landing_cross" and conf > threshold:#todo需要根据比赛具体细化
        if conf > threshold:

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 直接获取xyxy格式坐标
            x1_orig, y1_orig, x2_orig, y2_orig = map_to_original_coordinates(# 映射回原始坐标
                x1, y1, x2, y2, 
                original_width, original_height, original_width//4, original_height//4
            )
            
            # 计算中心点
            center_x = (x1_orig + x2_orig) // 2
            center_y = (y1_orig + y2_orig) // 2
            
            cv2.rectangle( 
                param.d435i_color, 
                (x1_orig, y1_orig), 
                (x2_orig, y2_orig), 
                (0, 255, 0), 2
            )
            """返回值格式[c_x,c_y,x1,y1,x2,y2,conf,class_id]"""
            return (
                center_x, center_y,
                x1_orig, y1_orig, 
                x2_orig, y2_orig, 
                conf, class_id 
            )
    
    return None


def map_to_original_coordinates(x1, y1, x2, y2, original_width, original_height, resized_width, resized_height):
    # 计算缩放因子
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    # 将缩放后的边界框坐标映射回原始图像的坐标
    x1_original = int(x1 * scale_x)
    y1_original = int(y1 * scale_y)
    x2_original = int(x2 * scale_x)
    y2_original = int(y2 * scale_y)
    
    return x1_original, y1_original, x2_original, y2_original
