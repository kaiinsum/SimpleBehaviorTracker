# 🎯 SimpleBehaviorTracker

A real-time focus tracking system built with **Python, OpenCV, MediaPipe, and NumPy**.  
This project monitors user attention through webcam analysis, including eye tracking, gaze detection, face orientation, and mouth movement detection.

## 📹 Demo
Introduction Video: https://www.youtube.com/watch?v=3qizNjBoRp8

---

## ⚙️ Installation

### First Run
Install required packages:

```bash
pip install opencv-python mediapipe numpy simpleaudio pygetwindow pyautogui matplotlib
```

### Requirements
- Python **3.11.0**
- Microsoft Visual C++ **14.0 or higher**

Install C++ Build Tools:  
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Required components:
- **C++ Build Tools**
- **MSVC v142 - VS 2019 C++ x64/x86 build tools**

---

# 🇻🇳 Hướng dẫn sử dụng (Tiếng Việt)

## Cách sử dụng
### 1. Setup ban đầu
- Giữ yên khuôn mặt.
- Nhìn tập trung vào dấu **"+"** giữa màn hình trong **5 giây** để hiệu chỉnh.

### 2. Theo dõi tập trung
Sau khi setup hoàn tất:
- Hãy duy trì sự tập trung vào màn hình.
- Nếu mất tập trung quá lâu, hệ thống sẽ phát cảnh báo.

Các hành vi được xem là mất tập trung:
- Ngáp ngủ
- Mắt lệch khỏi màn hình quá nhiều
- Quay đầu sang hướng khác
- ...

### 3. Điều khiển
Các phím chức năng:
- **I** → Bật/tắt thông tin phân tích
- **F** → Bật/tắt face mesh
- **R** → Reset / hiệu chỉnh lại

## Lưu ý
- Không thay đổi góc camera sau khi setup.
- Không di chuyển mặt quá nhiều trong 5 giây đầu.
- Luôn đảm bảo camera nhìn thấy toàn bộ khuôn mặt.
- Không che mặt để hệ thống hoạt động tốt nhất.

---

# 🇬🇧 User Guide (English)

## Instructions
### 1. Initial Setup
- Keep your face still.
- Focus on the **"+"** symbol at the center of the screen for **5 seconds**.

### 2. Focus Tracking
After setup:
- Maintain attention on your screen.
- If distraction is detected for too long, the system will trigger a warning.

Behaviors considered distracted:
- Yawning
- Looking too far away from screen
- Turning head away
- ...

### 3. Controls
Keyboard shortcuts:
- **I** → Toggle analysis information
- **F** → Toggle face mesh
- **R** → Reset calibration

## Notes
- Do not adjust camera angle after setup.
- Avoid excessive movement during setup.
- Keep your full face visible to camera.
- Avoid covering facial features.

---

# 🇨🇳 使用说明（简体中文）

## 操作步骤
### 1. 初始设置
- 保持面部静止。
- 在 **5 秒** 内注视屏幕中央的 **"+"** 符号。

### 2. 专注检测
设置完成后：
- 保持专注于电脑屏幕。
- 如果长时间分心，系统将发出警告。

以下行为会被判定为分心：
- 打哈欠
- 眼睛偏离屏幕过远
- 转头看向其他方向
- ……

### 3. 快捷键
- **I** → 开启/关闭分析信息
- **F** → 开启/关闭面部网格
- **R** → 重置校准参数

## 注意事项
- 设置完成后请勿调整摄像头角度。
- 设置期间避免大幅移动。
- 请确保摄像头完整捕捉面部。
- 请勿遮挡面部。

---

## 🛠 Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- PyAutoGUI
- SimpleAudio

---

## Features
✅ Eye tracking  
✅ Gaze detection  
✅ Yawning detection  
✅ Face direction tracking  
✅ Focus score monitoring  
✅ Warning alert system  
