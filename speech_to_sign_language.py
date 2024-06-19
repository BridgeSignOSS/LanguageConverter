import speech_recognition as sr
import nltk
import cv2
import os

nltk.download('punkt')

# 음성을 텍스트로 변환하는 함수
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio, language='ko-KR')
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return ""

# 텍스트 토큰화
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# 수어 동작 매핑
sign_language_map = {
    'Where does it hurt?': 'hurt.mp4',
    'No fever': 'no_fever.mp4',
    'High fever': 'high_fever.mp4',
    'Name and birthdate': 'name_date.mp4',
    'Administer anesthesia': 'administer_anesthesia.mp4',
    'After meals': 'after_meals.mp4',
    'Before bed': 'before_bed.mp4',
    'Before meals': 'before_meals.mp4',
    'Check blood pressure': 'check_blood_pressure.mp4',
    'Check temperature': 'check_temperature.mp4',
    'Evening': 'evening.mp4',
    'Fine': 'fine.mp4',
    'Get injection': 'get_injection.mp4',
    'Headache': 'headache.mp4',
    'Lunchtime': 'lunchtime.mp4',
    'Morning': 'morning.mp4',
    'Need surgery': 'need_surgery.mp4',
    'Normal': 'normal.mp4',
    'Sign surgical consent': 'sign_surgical_consent.mp4'

}

# 텍스트를 수어 동작으로 변환
def text_to_sign_language(text):
    tokens = tokenize_text(text)
    sign_videos = []
    for token in tokens:
        video = sign_language_map.get(token)
        if video:
            sign_videos.append(video)
        else:
            sign_videos.append(f"No sign video for '{token}'")
    return sign_videos

# 비디오 재생 함수
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Sign Language Video', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 함수
def main():
    # Step 1: 음성을 텍스트로 변환
    text = recognize_speech()
    
    if text:
        # Step 2: 인식된 텍스트를 수어 동작으로 변환
        sign_videos = text_to_sign_language(text)
        print("Sign language videos for the recognized text:")
        
        # Step 3: 동영상 재생
        for video in sign_videos:
            if "No sign video" in video:
                print(video)
            else:
                print(f"Playing video: {video}")
                play_video(video)

if __name__ == "__main__":
    main()
