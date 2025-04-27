import pyrealsense2 as rs2
import numpy as np
import cv2

class D435iRealtimeTester:
    def __init__(self):
        # 初始化相机参数（使用您提供的参数）
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = 640
        self.intrinsics.height = 480
        self.intrinsics.ppx = 326.437255859375
        self.intrinsics.ppy = 251.3828125
        self.intrinsics.fx = 608.034423828125
        self.intrinsics.fy = 607.7110595703125
        self.intrinsics.model = rs2.distortion.brown_conrady  # 使用的畸变模型
        self.intrinsics.coeffs = [0.0]*5
        
        # 初始化RealSense管道
        self.pipeline = rs2.pipeline()
        config = rs2.config()
        config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 30)
        config.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 30)
        self.pipeline.start(config)
        
        # 创建对齐器（彩色与深度对齐）
        self.align = rs2.align(rs2.stream.color)

    def run_test(self):
        try:
            while True:
                # 获取对齐后的帧
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # 显示彩色图并获取点击点
                cv2.imshow("Color Preview", color_image)
                key = cv2.waitKey(1)
                
                # 鼠标点击测试
                if key == ord('c'):
                    cv2.setMouseCallback("Color Preview", self.on_mouse_click, 
                                        (depth_image, color_image.copy()))
                
                elif key == 27:  # ESC退出
                    break
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def on_mouse_click(self, event, x, y, flags, param):
        """ 鼠标点击回调函数 """
        if event != cv2.EVENT_LBUTTONDOWN:
            return
            
        depth_image, color_image = param
        z = depth_image[y, x]/1000.0  # 转换为厘米
        
        # 坐标转换
        camera_coord = rs2.rs2_deproject_pixel_to_point(
            self.intrinsics, [x, y], z)
        yaw = 45 * np.pi / 180
        X0 = np.array([camera_coord[1], camera_coord[2]])
        R = np.array([[np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
        X_trans = np.dot(R, X0)
        print(X_trans)
        real_x_int,real_y_int,real_z_int = -X_trans[1]-0.337 , camera_coord[0], -X_trans[0]

        # 在图像上标记结果
        cv2.circle(color_image, (x,y), 5, (0,255,0), -1)

        # 打印控制台输出
        print("\n" + "="*50)
        print(f"像素坐标: ({x}, {y})")
        print(f"深度值: {z:.3f} m")
        print(f"相机坐标系: X={camera_coord[0]:.3f}m, Y={camera_coord[1]:.3f}m, Z={camera_coord[2]:.3f}m")
        print(f"转换后T265相机坐标系: X={real_x_int:.3f}m, Y={real_y_int:.3f}m, Z={real_z_int:.3f}m")
        # 显示结果
        cv2.imshow("Click Result", color_image)

if __name__ == "__main__":
    tester = D435iRealtimeTester()
    print("操作说明:")
    print("1. 按 'c' 键启用点击测试模式")
    print("2. 在彩色图像上点击目标点")
    print("3. 按 ESC 退出程序")
    tester.run_test()