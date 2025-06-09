import pyrealsense2 as rs
import numpy as np
import cv2

##########################王者荣耀
def houf_circle(frame, dp, minDist, param1, param2, minRadius, maxRadius):
    """
    霍夫圆检测
    输入frame
    输出圆心坐标
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    # 直方图均衡化
    equlized_image = cv2.equalizeHist(blur_image)
    equlized_image = cv2.bilateralFilter(equlized_image, 9, 100, 100)  # d=9, sigmaColor=75, sigmaSpace=75
    circles = cv2.HoughCircles(equlized_image, cv2.HOUGH_GRADIENT_ALT, dp=dp,
                               minDist=minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)  # 改进的霍夫梯度

    if circles is not None:
        circles = circles[0, :, :]
        return circles
    else:
        return None

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create a window for displaying the image and sliders
    cv2.namedWindow('Circle Detection')
    dp = 1
    minDist = 399
    param1 = 73
    param2 = 0.78
    minRadius = 60
    maxRadius = 436

    cv2.createTrackbar('dp', 'Circle Detection', dp, 10, lambda x: None)
    cv2.createTrackbar('minDist', 'Circle Detection', minDist, 1000, lambda x: None)
    cv2.createTrackbar('param1', 'Circle Detection', param1, 255, lambda x: None)
    cv2.createTrackbar('param2', 'Circle Detection', int(param2 * 100), 100, lambda x: None)
    cv2.createTrackbar('minRadius', 'Circle Detection', minRadius, 1000, lambda x: None)
    cv2.createTrackbar('maxRadius', 'Circle Detection', maxRadius, 1000, lambda x: None)
    cap = cv2.VideoCapture(0)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            _,color_image = cap.read()
            # Get current values from sliders
            dp = cv2.getTrackbarPos('dp', 'Circle Detection')
            minDist = cv2.getTrackbarPos('minDist', 'Circle Detection')
            param1 = cv2.getTrackbarPos('param1', 'Circle Detection')
            param2 = cv2.getTrackbarPos('param2', 'Circle Detection') / 100.0
            minRadius = cv2.getTrackbarPos('minRadius', 'Circle Detection')
            maxRadius = cv2.getTrackbarPos('maxRadius', 'Circle Detection')

            # Perform circle detection
            circles = houf_circle(color_image, dp, minDist, param1, param2, minRadius, maxRadius)

            if circles is not None:
                for c in circles:
                    cv2.circle(color_image, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 2)
                    cv2.circle(color_image, (int(c[0]), int(c[1])), 2, (0, 0, 0), -1)

            # Display the resulting frame
            cv2.imshow('Circle Detection', color_image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()