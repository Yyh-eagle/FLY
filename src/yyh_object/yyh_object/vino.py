import cv2
import numpy as np
from openvino.runtime import Core
import time
#------------------------------------------------------------#
#  本文件用于使用神经网络量化推理加速
#------------------------------------------------------------#
SIZE = 320

#初始化 
def Init_ie():
    model_xml = "/home/yyh/ros2_ws/src/yyh_object/yyh_object/best.xml"
    ie = Core()
    model = ie.read_model(model=model_xml)
    #----------------------------------------------------#
    #    需要给推理时固定输入大小，否则无法正确推理。
    #----------------------------------------------------#
    
    input_layer = model.inputs[0]
    print("Model input partial shape:", input_layer.partial_shape)
    model.reshape({input_layer.any_name: [1, 3, SIZE, SIZE]})

    #----------------------------------------------------#
    #    编译模型到CPU中
    #----------------------------------------------------#
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    infer_request = compiled_model.create_infer_request()#创建推理请求对象。
    return input_layer, infer_request,model
    
def yolo_int8(frame,threshold,input_layer,infer_request,model):#todo应该将初始化函数和推理函数分开

   
    #----------------------------------------------------#
    #   从param中取出参数
    #----------------------------------------------------#

    #----------------------------------------------------#
    #   预处理、推理、后处理
    #----------------------------------------------------#
    input_tensor, orig_frame = preprocess_frame(frame, input_layer.partial_shape)#数据预处理
    results = infer_request.infer({input_layer.any_name: input_tensor})#得到推理结果results
    output_layer = model.outputs[0]
    output_data = results[output_layer.any_name]#获取输出数据                                                                     
    detections = postprocess(output_data, conf_threshold=0.5)

    orig_h, orig_w = orig_frame.shape[:2]
    
    scale_x = orig_w / SIZE
    scale_y = orig_h / SIZE

    for det in detections:#todo 当前代码不具有处理多目标的情况
        bbox = det['bbox']
        bbox[0] = int(bbox[0] * scale_x)
        bbox[2] = int(bbox[2] * scale_x)
        bbox[1] = int(bbox[1] * scale_y)
        bbox[3] = int(bbox[3] * scale_y)
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        cv2.rectangle( 
            frame, 
            (bbox[0], bbox[1]), 
            (bbox[2], bbox[3]), 
            (0, 255, 0), 2
        )
        conf = float(det['confidence'])
        class_id = det['class_id']
        """返回值格式[c_x,c_y,x1,y1,x2,y2,conf,class_id]"""
        return (
            center_x, center_y,
            bbox[0], bbox[1], 
            bbox[2], bbox[3], 
            conf, class_id 
        )

 




#图像预处理
def preprocess_frame(frame, input_shape):
    h, w = input_shape[2].get_length(), input_shape[3].get_length()
    if h is None or w is None:
        raise ValueError("Input shape height or width is dynamic, cannot resize image.")
    resized = cv2.resize(frame, (w, h))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = resized.astype(np.float32) / 255.0
    input_tensor = resized.transpose((2, 0, 1))[np.newaxis, ...]
    return input_tensor, frame

#图像后处理
def postprocess(output_data, conf_threshold=0.6):
    detections = output_data[0]
    results = []
    for det in detections:
        x_min, y_min, x_max, y_max, conf, class_id = det
        if conf > conf_threshold:
            results.append({
                'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                'confidence': conf,
                'class_id': int(class_id)
            })
    return results

