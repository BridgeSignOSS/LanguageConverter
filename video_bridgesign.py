import speech_recognition as sr
import os
from langchain_community.llms import Ollama
import nltk
import cv2

nltk.download('punkt')

# 동영상이 저장된 폴더 경로 설정
IMAGE_FOLDER = r'C:\Users\ihj05\python\HandGesture\mp4'

# 특정 단어들과 그에 해당하는 이미지 파일 이름들 매핑
sign_language_map = {
    'where does it hurt': 'hurt.mp4',
    'no fever': 'no_fever.mp4',
    'high fever': 'high_fever.mp4',
    'name and birthdate': 'name_date.mp4',
    'administer anesthesia': 'administer_anesthesia.mp4',
    'after meals': 'after_meals.mp4',
    'before bed': 'before_bed.mp4',
    'before meals': 'before_meals.mp4',
    'check blood pressure': 'check_blood_pressure.mp4',
    'check temperature': 'check_temperature.mp4',
    'evening': 'evening.mp4',
    'fine': 'fine.mp4',
    'get injection': 'get_injection.mp4',
    'headache': 'headache.mp4',
    'lunchtime': 'lunchtime.mp4',
    'morning': 'morning.mp4',
    'need surgery': 'need_surgery.mp4',
    'normal': 'normal.mp4',
    'sign surgical consent': 'sign_surgical_consent.mp4'
}

llm = Ollama(model="bridgesign:latest")

def get_model_response(user_prompt):
    response = llm.invoke(user_prompt)
    return response.strip()

# 음성 인식을 통해 텍스트 생성
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def tokenize_text(text):
    tokens = text.split(", ")
    return tokens

def text_to_sign_language(text):
    tokens = tokenize_text(text)
    sign_videos = []
    for token in tokens:
        video = sign_language_map.get(token)
        if video:
            sign_videos.append(os.path.join(IMAGE_FOLDER, video))
        else:
            sign_videos.append(f"No sign video for '{token}'")
    return sign_videos

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

if __name__ == "__main__":
    text = recognize_speech()
    if text:
        ai_response = get_model_response(text)
        if ai_response:
            print(f"AI response: {ai_response}")
            sign_videos = text_to_sign_language(ai_response)
            print("Sign language videos for the input text:")
            for video in sign_videos:
                if "No sign video" in video:
                    print(video)
                else:
                    print(f"Playing video: {video}")
                    play_video(video)
