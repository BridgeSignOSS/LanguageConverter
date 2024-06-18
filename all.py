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


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 데이터 로드
landmarks_data = np.load('landmarks_data.npy')
labels = np.load('labels.npy')

# 라벨 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(landmarks_data, labels_encoded, test_size=0.2, random_state=42)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 모델 저장
model.save('gesture_recognition_model.h5')

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
    1: 'My name is Eunsoo',
    2: 'Nice to meet you'
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
