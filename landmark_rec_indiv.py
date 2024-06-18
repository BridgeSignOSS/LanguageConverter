import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장을 위한 리스트
landmarks_data = []
labels = []

# 비디오 파일 리스트와 해당 라벨
video_files = [
    ('hello.mp4', 'Hello'),
    ('my_name_is_eunsoo.mp4', 'My name is Eunsoo'),
    ('nice_to_meet_you.mp4', 'Nice to meet you')
]

for video_file, label in video_files:
    cap = cv2.VideoCapture(video_file)
    
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
                labels.append(label)
    
    cap.release()

hands.close()

# numpy 배열로 변환하여 저장
landmarks_data = np.array(landmarks_data)
labels = np.array(labels)

np.save('landmarks_data.npy', landmarks_data)
np.save('labels.npy', labels)
