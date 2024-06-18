pip install opencv-python mediapipe tensorflow
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# MediaPipe 손 인식 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 손 제스처 분류 모델 로드 (사전 훈련된 모델 사용)
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# 제스처 레이블 정의
gesture_labels = {
    0: 'Hello',
    1: 'Yes',
    2: 'No',
    3: 'Thank You'
}

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임을 BGR에서 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 인식 수행
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 랜드마크 좌표를 리스트로 변환
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            # 모델 입력을 위해 좌표 정규화 및 변환
            landmarks = np.array(landmarks).flatten().reshape(1, -1)
            
            # 제스처 예측
            prediction = model.predict(landmarks)
            class_id = np.argmax(prediction)
            gesture_text = gesture_labels[class_id]
            
            # 예측된 제스처를 프레임에 텍스트로 표시
            cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 프레임 출력
    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
hands.close()
