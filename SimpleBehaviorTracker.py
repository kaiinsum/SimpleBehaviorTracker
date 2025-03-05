"""
Introduction: https://www.youtube.com/watch?v=3qizNjBoRp8

First run:
Window -> cmd -> (python11.3 -m) pip install opencv-python mediapipe numpy simpleaudio pygetwindow pyautogui matplotlib

Requirements:
Python 3.11.0
Microsoft Visual C++ 14.0 or greater. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
C++ Build Tools -> MSVC v142 - VS 2019 C++ x64/x86 build tools

-----------------------------------------------------------------------------------------------------------
>=============================================< TIẾNG VIỆT >==============================================<

    Hướng dẫn:
    1. Giữ yên khuôn mặt và mắt tập trung vào dấu cộng giữa màn hình trong 5 giây set up.
    2. Sau 5 giây set up, hãy duy trì sự tập trung vào màn hình máy tính, nếu mất tập trung quá lâu máy sẽ đưa ra cảnh báo.
    Những hành động được tính là mất tập trung:
        - Ngáp ngủ
        - Di chuyển mắt lệch khỏi màn hình quá nhiều
        - Quay sang hướng khác
        - ...
    3. Sau 5 giây set up, bạn có quyền điều chỉnh với các nút sau:
    - I (info): Bật/tắt hiển thị thông số phân tích.
    - F (face mesh): Bật/tắt hiển thị mặt nạ.
    - R (reload/reset): Đặt lại thông số hiệu chỉnh.

    * Lưu ý:
    1. Không điều chỉnh góc camera trong và sau khi đã set up xong.
    2. Không di chuyển mặt quá nhiều trong 5 giây set up.
    3. Trong suốt quá trình chương trình chạy, hãy luôn để camera có thể theo dõi toàn bộ khuôn mặt của bạn, không bị cắt xén.
    4. Hãy để mọi thứ tự nhiên, không che phần khuôn mặt để chương trình chạy với hiệu suất tốt nhất.

>===============================================< ENGLISH >================================================<

    Instruction:
    1. Keep your face still and focus your eyes on the cross in the center of the screen for 5 seconds during setup.
    2. After the 5-second setup, maintain focus on the computer screen. If you lose focus for too long, the system will issue a warning.
    Actions considered as loss of focus:
        - Yawning
        - Moving your eyes too far away from the screen
        - Turning your head in another direction
        - ...
    3. After the 5-second setup, you can adjust the following settings using these keys:
    - I (info): Turn on/off analysis information display.
    - F (face mesh): Turn on/off face mask display.
    - R (reload/reset): Reset adjustment parameters.

    * Notes:
    1. Do not adjust the camera angle during or after setup.
    2. Avoid excessive head movement during the 5-second setup.
    3. Throughout the program's operation, ensure that your entire face remains visible to the camera without being cropped.
    4. Keep everything natural and avoid covering your face to ensure optimal program performance.

>==============================================< 中文（简体) >===============================================<

    使用指南：
    1. 在 5 秒设置 期间，保持面部静止，眼睛专注于屏幕中央的“+”符号。
    2. 5 秒设置 结束后，请继续专注于电脑屏幕，如果长时间分心，系统将发出警告。
    以下行为将被视为分心：
        - 打哈欠
        - 眼睛偏离屏幕太多
        - 转头看向其他方向
        - ……
    3. 5 秒设置 结束后，您可以使用以下按键进行调整：
    - I (信息）：开启/关闭分析数据显示。
    - F (面部网格）：开启/关闭面部网格显示。
    - R (重置）：重置调整参数。

    * 注意事项：
    1. 设置完成后，请勿调整摄像头角度。
    2. 在 5 秒设置 期间，请勿大幅度移动面部。
    3. 运行程序时，请确保摄像头能完整捕捉您的面部，不被遮挡或裁剪。
    4. 请保持自然状态，不要遮挡面部，以确保程序最佳运行效果。
"""

import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import ctypes
import pygetwindow as gw
import pyautogui
import matplotlib.pyplot as plt
from collections import deque

# var
window_title = "BEHAVIOR TRACKER"

soundclock=0
clock = 0
chopmat = 0
credit=40
prevavgeye1=0
prevavgeye2=0
tickdelay=0
seconddelay=0
ifchopmat = False
ifngap=False
ifsound=False
countxuong=0
lefteye_test=1
righteye_test=1
countrgb=0
lefteye_start=0
righteye_start=0
lostfocuscount=0
chopchop=0
intro = sa.WaveObject.from_wave_file("./audio/intro.wav")
wave_obj = sa.WaveObject.from_wave_file("./audio/alarm.wav")  

# setting
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
MAX_NUM_FACES = 1
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("my_app_id")

# def
def set_window_icon(icon_path):
    try:
        win = gw.getWindowsWithTitle(window_title)
        if win:
            hwnd = win[0]._hWnd
            ctypes.windll.user32.SendMessageW(hwnd, 0x80, 1, icon_path)
    except Exception as e:
        print("Không thể đặt icon:", e)

def eye_aspect_ratio(landmarks, eye_points):
    A = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
    B = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
    C = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
    return (A + B) / (2.0 * C)

def detect_mouth_expression(landmarks, mouth_points):
    hull = cv2.convexHull(landmarks[mouth_points])
    x ,y, w, h = cv2.boundingRect(hull) 
    return h / w  

def detect_gaze_direction(landmarks, eye_points, pupil_point):
    left, right = landmarks[eye_points[0]], landmarks[eye_points[3]]  
    pupil = landmarks[pupil_point]  
    eye_width = right[0] - left[0]  
    pupil_pos = (pupil[0] - left[0]) / eye_width 
    return pupil_pos

# anchoring
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 13, 310, 311, 312, 82, 81, 80, 191]
LEFT_PUPIL = 468
RIGHT_PUPIL = 473

cap = cv2.VideoCapture(0)

# show
show_coordinates = True
show_face_mesh = False 
show_info = False

with mp_face_mesh.FaceMesh(
    max_num_faces=MAX_NUM_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        info_panel = np.zeros((h, 300, 3), dtype=np.uint8)

        # if face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if show_face_mesh:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

                landmarks = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])
                if show_face_mesh:
                    for idx in LEFT_EYE:
                        cv2.circle(image, tuple(landmarks[idx]), 2, (255, 0, 0), -1)
                    for idx in RIGHT_EYE:
                        cv2.circle(image, tuple(landmarks[idx]), 2, (255, 0, 0), -1)
                    for idx in MOUTH:
                        cv2.circle(image, tuple(landmarks[idx]), 2, (0, 255, 0), -1)
                    cv2.circle(image, tuple(landmarks[LEFT_PUPIL]), 3, (0, 0, 255), -1)
                    cv2.circle(image, tuple(landmarks[RIGHT_PUPIL]), 3, (0, 0, 255), -1)
                
                mouth_expansion = detect_mouth_expression(landmarks, MOUTH)
                if ifsound==True and credit<50 and soundclock <=100:
                    soundclock+=1
                    if (soundclock == 100):
                        lostfocuscount+=1
                        sound_playing = wave_obj.play()
                        print("playsound")
                        ifsound = False
                elif soundclock < 200 and soundclock>=100:
                    soundclock+=1
                    ifsound = False
                    if soundclock == 199:
                        soundclock = 0
                else: 
                    soundclock=0
                
                if mouth_expansion > 0.8:
                    print("Ngáp")
                    if credit>4:
                        credit-=(1.001**(100-credit)*0.001*credit**2)*0.91
                    ifngap = False
                else:
                    ifngap = True
                
                # eye alg
                if lefteye_test<1:
                    left_gaze = detect_gaze_direction(landmarks, LEFT_EYE, LEFT_PUPIL)/abs(lefteye_test)
                elif lefteye_test>1:
                    left_gaze = detect_gaze_direction(landmarks, LEFT_EYE, LEFT_PUPIL)*abs(lefteye_test)
                else:
                    left_gaze = detect_gaze_direction(landmarks, LEFT_EYE, LEFT_PUPIL)
                if righteye_test<1:
                    right_gaze = detect_gaze_direction(landmarks, RIGHT_EYE, RIGHT_PUPIL)/abs(righteye_test)
                elif righteye_test>1:
                    right_gaze = detect_gaze_direction(landmarks, RIGHT_EYE, RIGHT_PUPIL)*abs(righteye_test)
                else:
                    right_gaze = detect_gaze_direction(landmarks, RIGHT_EYE, RIGHT_PUPIL)
                left_eye = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_eye = eye_aspect_ratio(landmarks, RIGHT_EYE)
                avg_eye = (left_eye + right_eye) / 2.0
                
                if left_gaze < 0.35:
                    print("Liếc phải")
                    if credit > 6:
                        credit-= abs(0.35-left_gaze)*0.07*abs(credit-100)
                elif right_gaze > 0.65:
                    print("Liếc trái")
                    if credit > 6:
                        credit-= abs(0.65-right_gaze)*0.07*abs(credit-100)
                elif not (right_gaze > 0.65 and left_gaze < 0.35):
                    if avg_eye > 0.29 and right_eye > 0.29 and left_eye > 0.29:
                        print("Liếc lên")
                        if credit > 5:
                            credit-= abs(abs(0.31-avg_eye)*credit*5.3-(credit/20))
                    elif avg_eye < 0.2 and right_eye < 0.2 and left_eye < 0.2 and ifchopmat==True:
                        countxuong+=1
                        if countxuong>20:
                            if credit > 6:
                                credit-= abs(abs(0.22-avg_eye)*credit*0.2*countxuong-(credit/30))*0.04
                            print("Liếc xuống")
                    elif countxuong>0:
                        countxuong-=1
        
                combined_image = np.hstack((image, info_panel))
                
                if show_info:
                    cv2.putText(info_panel, f"Left EYE: {left_eye:.2f} + Gaze: {left_gaze:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(info_panel, f"Right EYE: {right_eye:.2f} + Gaze: {right_gaze:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(info_panel, f"Avg EYE: {avg_eye:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(info_panel, f"Mouth: {mouth_expansion:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(info_panel, f"T.L-Eye {lefteye_test:.2f} | T.R-Eye {righteye_test:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    combined_image = np.hstack((image, info_panel))
                if show_coordinates:
                    for idx, (x, y) in enumerate(landmarks):
                        cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.16, (0, 255, 255), 1)
                        
                if credit <=100 and credit >=0:
                    prevavgeye2=prevavgeye1
                    prevavgeye1=avg_eye
                    if prevavgeye2 < avg_eye and credit > 5:
                        credit-=1.5*(avg_eye-prevavgeye2)
                    else:
                        if credit < 92:
                            credit+=abs(avg_eye-prevavgeye2)*(100-credit)*0.035+(100-credit)/(21*2.1)
                        else:
                            credit+=abs(avg_eye-prevavgeye2)*(100-credit)*0.015+(100-credit)/(21*5.1)

                if credit<50:
                    ifsound=True
                if credit<0:
                    credit=0
                
                if left_eye > 0.17 and right_eye < 0.17:
                    print("Chớp mắt trái")
                    if credit > ((1.1**((100-credit)*-0.02))/1.3+5):
                        credit-=(1.1**((100-credit)*-0.02))/1.3
                    ifchop1mat=True
                if left_eye < 0.17 and right_eye > 0.17:
                    print("Chớp mắt phải")
                    if credit > ((1.1**((100-credit)*-0.02))/1.3+5):
                        credit-=(1.1**((100-credit)*-0.02))/1.3
                    ifchop1mat=True
                else:
                    ifchop1mat=False
                if avg_eye < 0.17 and ifchop1mat==False and left_eye < 0.08 and right_eye < 0.08:
                    clock += 1
                    if ifchopmat == True and ifngap == True:
                        chopmat += 1
                        print("Chớp 2 mắt")
                        if credit > (1.1**((100-credit)*-0.02)+1):
                            credit-=1.1**((100-credit)*-0.02)
                        ifchopmat = False
                        countxuong=0
                else:
                    if not (seconddelay<=5 and tickdelay<32):
                        ifchopmat = True
                        if soundclock>=100:
                            chopchop+=1
                            if chopchop <=10:
                                cv2.putText(combined_image, f"Xin vui long tap trung", (30, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                                cv2.putText(combined_image, f"Ban dang danh mat tuong lai cua chinh minh", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                            elif chopchop ==15:
                                chopchop=0
                        else:
                            chopchop=0
                        cv2.putText(combined_image, f"So lan mat tap trung", (w + 10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(combined_image, f"(lost focus count): {lostfocuscount:.0f}", (w + 10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)                        
                        cv2.putText(combined_image, f"% Focus: {credit:.2f}", (w + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        #cv2.putText(combined_image, f"Test: {soundclock}", (w + 20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        #cv2.putText(combined_image, f"Tgian: {soundclock} Lan: {chopmat}", (w + 20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # intro time
                    if seconddelay<=5 and tickdelay<33:
                        credit=63
                        tickdelay+=1
                        if seconddelay>=3:
                            for i in range(1,30*3+1,1):
                                lefteye_start=(lefteye_start*(i-1)+left_gaze)/i
                                righteye_start=(righteye_start*(i-1)+right_gaze)/i
                        if tickdelay==30 and seconddelay<5:
                            seconddelay+=1
                            tickdelay=0
                        if tickdelay==1 and seconddelay==0:
                            sound_playing = intro.play()
                            
                        # personal adj
                        if tickdelay==32:
                            lefteye_test=(0.447/lefteye_start - (0.447-lefteye_start))*0.97
                            righteye_test=(0.582/righteye_start - (0.582-righteye_start))*0.97
                            print(righteye_start)
                            print(lefteye_start)
                            show_info = True
                            show_face_mesh = True
                        """if (5-seconddelay)==1:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        elif (5-seconddelay)==2:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                        elif (5-seconddelay)==3:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        elif (5-seconddelay)==0:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)"""
                        
                        cv2.putText(combined_image, f"Vui long ngoi dung tu ", (w+5 , 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(combined_image, f"the va nhin vao giua", (w+5, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(combined_image, f"dau cong trong ", (w+5, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        if (5-seconddelay)<=1:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (w+213, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        elif (5-seconddelay)<=2:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (w+213, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        elif (5-seconddelay)<=3:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (w+213, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(combined_image, f"{(5-seconddelay):.0f} ", (w+213, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(combined_image, f"  giay", (w+215, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        countrgb+=1
                        if countrgb<10:
                            cv2.putText(combined_image, f"+", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                        elif countrgb<20:
                            cv2.putText(combined_image, f"+", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                        elif countrgb<30:
                            cv2.putText(combined_image, f"+", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                        elif countrgb<40:
                            cv2.putText(combined_image, f"+", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                        else:
                            countrgb=0
                    
        else:
            # if not face
            if credit>0:
                credit-=(credit)*0.003*(100-credit)
            combined_image = np.hstack((image, info_panel))
            cv2.putText(combined_image, f"%Focus: {credit:.2f}", (w + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            #cv2.putText(combined_image, f"Tgian: {countxuong} Lan: {chopmat}", (w + 20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(combined_image, "Vui long de mat vao", (w + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(combined_image, "camera!", (w + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #cv2.putText(combined_image, f"{lefteye_start:.2f} + {righteye_start}", (w + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #cv2.putText(combined_image, f"{lefteye_test:.2f} + {righteye_test}", (w + 20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            clock+=1       
            clock+=1       
                
        set_window_icon("./img/icon.ico")
        cv2.imshow(window_title, combined_image)

        # key
        key = cv2.waitKey(5) & 0xFF
        if key == 27: 
            break
        elif (key == ord('r') or key == ord('R')) and tickdelay>31:
            seconddelay=0
            tickdelay=0
            lefteye_start=0
            righteye_start=0
            show_face_mesh = False
            show_info = False
        elif key == ord('i') or key == ord('I'):  
            show_info = not show_info
        elif key == ord('c') or key == ord('C'):  
            show_coordinates = not show_coordinates
        elif key == ord('f') or key == ord('F'): 
            show_face_mesh = not show_face_mesh
        elif key == ord('q') or key == ord('Q'):
            if sound_playing:
                sound_playing.stop()
                sound_playing = None
            else:
                sound_playing = wave_obj.play()

cap.release()
cv2.destroyAllWindows()
