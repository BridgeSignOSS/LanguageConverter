import nltk
import cv2
import os

nltk.download('punkt')

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# 수어 동작 매핑
sign_language_map = {
    'hospital': 'hospital_sign.mp4'
}

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

def main():
    text = input("Enter text to convert to sign language: ")
    sign_videos = text_to_sign_language(text)
    print("Sign language videos for the input text:")
    for video in sign_videos:
        if "No sign video" in video:
            print(video)
        else:
            print(f"Playing video: {video}")
            play_video(video)

if __name__ == "__main__":
    main()
