import speech_recognition as sr
import cv2
from langchain_community.llms import Ollama
from langchain import PromptTemplate

# 이미지가 저장된 폴더 경로 설정
IMAGE_FOLDER = r'C:\Users\dmstn\OneDrive\바탕 화면\openss_team\image'

# 특정 단어들과 그에 해당하는 이미지 파일 이름들 매핑
word_to_image = {
    'hospital': 'hospital_sign.mp4',
    'check': {
        'blood pressure': 'check_blood_pressure.mp4',
        'temperature': 'check_temperature.mp4'
    },
    'evening': 'evening.mp4',
    'fine': 'fine.mp4',
    'morning': 'morning.mp4',
    'before_bed': 'before_bed.mp4',
    'headache': 'headache.mp4',
    'injection': 'get_injection.mp4',
    'normal': 'normal.mp4',
    'after meals': 'after_meals.mp4'
}

def display_image(image_path):
    img = cv2.imread(image_path)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)  # 최대 5초 동안 대기
        except sr.WaitTimeoutError:
            print("Timeout error occurred. Listening timed out.")
            return ""

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio, language='en-US')  # 영어 (미국)으로 설정
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service: {e}")
        return ""

llm = Ollama(model="llama3", stop=[""])

def get_model_response(user_prompt, system_prompt):
    template = """
    system
    {system_prompt}
    
    user
    {user_prompt}
    
    <|start_header_id>assistant
    """

    prompt = PromptTemplate(
        input_variables=["system_prompt", "user_prompt"],
        template=template
    )

    try:
        response = llm(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
        return response
    except Exception as e:
        print(f"Error occurred during model response generation: {e}")
        return None

def main():
    # Step 1: 음성을 텍스트로 변환
    user_prompt_ = recognize_speech()

    if user_prompt_:
        # word_to_image에서 매핑된 이미지 파일 이름을 찾아서 시스템 프롬프트에 삽입
        system_prompt_ = f"I'm currently working on extracting words. The list of words I have is {[word for word in word_to_image.keys()]}. Using these words, please rephrase the sentence the doctor said. Just send me that sentence."
        final_text = get_model_response(system_prompt_, user_prompt_)

        if final_text:
            # Step 2: 인식된 텍스트 출력
            print(f"Recognized Text: {final_text}")

if __name__ == "__main__":
    main()
