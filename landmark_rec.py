import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 읽기
cap = cv2.VideoCapture('your_video_file.mp4')
landmarks_data = []
labels = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR에서 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 인식 수행
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 랜드마크 좌표 추출
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks_data.append(landmarks)
            # 각 프레임에 맞는 라벨 추가 (예: 'Hello')
            labels.append('Hello')
    
cap.release()
hands.close()

# numpy 배열로 변환
landmarks_data = np.array(landmarks_data)
labels = np.array(labels)
