import speech_recognition as sr

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

def main():
    # Step 1: 음성을 텍스트로 변환
    text = recognize_speech()
    
    if text:
        # Step 2: 인식된 텍스트 출력
        print(f"Recognized Text: {text}")

if __name__ == "__main__":
    main()
