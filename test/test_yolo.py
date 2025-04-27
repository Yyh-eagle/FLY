import time
import cv2
import numpy as np
import pyrealsense2 as rs
from collections import deque
from ultralytics import YOLOv10  # 确保安装 ultralytics>=8.14.0

# 初始化RealSense相机
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 分辨率可调整
    pipeline.start(config)
    return pipeline

# 加载YOLOv10模型（根据实际模型路径修改）
def load_yolov10():
    model = YOLOv10("/home/yyh/ros2_ws/src/yyh_object/yyh_object/yolov10n.pt")  # 可选: yolov10s/m/l/x
    return model

# 主测试函数
def test_yolov10_fps():
    # 初始化
    pipeline = init_realsense()
    model = load_yolov10()
    fps_queue = deque(maxlen=30)  # 平滑FPS计算
    window_name = "YOLOv10 + D435i FPS Test"
    
    try:
        while True:
            start_time = time.perf_counter()
            
            # 1. 从D435i获取图像
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            frame = np.asanyarray(color_frame.get_data())  # 零拷贝转换
            
            # 2. YOLOv10推理
            results = model(frame, imgsz=640, verbose=False)  # 关闭冗余输出
            annotated_frame = results[0].plot()  # 绘制检测框
            
            # 3. 计算FPS
            fps = 1 / (time.perf_counter() - start_time)
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)
            
            # 显示结果
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, annotated_frame)
            
            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_yolov10_fps()