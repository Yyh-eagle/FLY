import cv2                             
import numpy as np                      


import pyrealsense2 as rs2#python内置的d435i库
from learning_interface.msg import ObjectPosition
from Coord_Trans import locate_d4
YOLO_THRESHOLD = 0.5
def yolo_root_d4(param,yolo):
    """
    yolo的总功能函数
    """

    result=yolo_recog_V10(param,YOLO_THRESHOLD,yolo)#threshold =0.8
    #param.logger.info(str(result))
    #result=yolo_recog_V5(param,YOLO_THRESHOLD,yolo)#threshold =0.8
    if result is None:
        return None
    return locate_d4(param,result) 
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


def yolo_recog_V10(param, threshold, yolo):
    color_d435i = param.d435i_color
    
    resized_image = cv2.resize(color_d435i, (424, 240))  # (width, height)
    # 使用缩放后的图像进行预测
    detect_result = yolo.predict(resized_image,imgsz=424,half=False,max_det=1,verbose=False)       # 关闭调试输出)  # 用缩放后的图像做预测
    #detect_result = yolo.predict(color_d435i)
    result = detect_result[0]  # 获取第一个结果（单图像输入）
    # 原始图像尺寸
    original_height, original_width = param.d435i_color.shape[:2]

      # 处理检测结果
    for box in result.boxes:
        cls_id = int(box.cls)
        name = result.names[cls_id]
        conf = float(box.conf)
        #if name == "landing_cross" and conf > threshold:
        if conf > threshold:

            # 获取缩放后坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 直接获取xyxy格式坐标
            
            # 映射回原始坐标
            x1_orig, y1_orig, x2_orig, y2_orig = map_to_original_coordinates(
                x1, y1, x2, y2, 
                original_width, original_height, 424, 240
            )
            
            # 计算中心点
            center_x = (x1_orig + x2_orig) // 2
            center_y = (y1_orig + y2_orig) // 2
            
            # 绘制检测框（调试用）
            cv2.rectangle(
                param.d435i_color, 
                (x1_orig, y1_orig), 
                (x2_orig, y2_orig), 
                (0, 255, 0), 2
            )
            
            return (
                center_x, center_y,
                x1_orig, y1_orig, 
                x2_orig, y2_orig, 
                conf, 1  # 保持与原代码相同的返回格式
            )
    
    return None


