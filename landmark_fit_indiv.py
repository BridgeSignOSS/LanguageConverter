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
