import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
i = 0
while True:

    # 读取视频流中的一帧
    ret, frame = cap.read()

    if not ret:
        print("无法获取视频流")
        break

    # 显示当前帧
    cv2.imshow('Camera Feed', frame)

    # 检查按键
    key = cv2.waitKey(1) & 0xFF

    # 如果按下空格键，保存图像
    if key == ord(' '):
        i += 1
        cv2.imwrite(f'saved_image{i}.jpg', frame)
        print(f"图像{i}已保存！")

    # 按下 'q' 键退出
    if key == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
