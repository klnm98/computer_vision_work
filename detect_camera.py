import cv2
import numpy as np

print("正在启动摄像头...")
cap = cv2.VideoCapture(0) # 0 代表默认摄像头

if not cap.isOpened():
    print("❌ 无法打开摄像头！")
    print("请检查：1. 微信/QQ是否占用了摄像头？ 2. Windows隐私设置是否允许桌面应用访问摄像头？")
else:
    print("✅ 摄像头已启动！将积木放在镜头前。按键盘上的 'q' 键可以退出程序。")
    
    while True:
        # 1. 读取一帧画面
        ret, frame = cap.read()
        if not ret:
            print("无法读取画面")
            break

        # 2. 图像预处理（为了让电脑更容易找到积木）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰度
        blur = cv2.GaussianBlur(gray, (5, 5), 0)       # 高斯滤波去噪
        
        # 3. 二值化（把背景和积木分离，自适应阈值）
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. 严格过滤轮廓（保证识别出积木不出错的核心）
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 规则1：面积过滤（根据摄像头距离调整，这里大概是一个巴掌大小的面积）
            if 3000 < area < 200000:
                peri = cv2.arcLength(cnt, True)
                # 规则2：多边形逼近，必须是4个角（积木是长方体）
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    
                    # 规则3：长宽比过滤（积木是正方形，不能太狭长）
                    if 0.7 < aspect_ratio < 1.3:
                        # 通过了所有严格测试，说明找到了积木！
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3) # 画绿框
                        cv2.putText(frame, "Block Detected", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 6. 显示结果
        cv2.imshow('Block Recognition', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出。")