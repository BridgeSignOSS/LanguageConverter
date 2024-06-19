import speech_recognition as sr
import nltk
import cv2
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

nltk.download('punkt')
# Set your Hugging Face token
os.environ['HF_TOKEN'] = "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"


def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
    )
    return tokenizer, model

def generate_response(system_message, user_message, tokenizer, model):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)

def summarize_text(text):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, model = load_model_and_tokenizer(model_id)
    
    system_message = (
        "You are a chatbot that performs summarization. Summarize the key content using "
        "one of the following words: Where does it hurt, No fever, High fever, "
        "Name and birthdate, Administer anesthesia, "
        "After meals, Before bed, Before meals, Check blood pressure, Check temperature, "
        "Evening, Fine, Get injection, Headache, Lunchtime, Morning, Need surgery, Normal, "
        "Sign surgical consent. Choose from these words to express it simply. If multiple are included, let me know."
    )
    
    return generate_response(system_message, text, tokenizer, model)

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
    
    speech_to_text = recognize_speech()
    original_text = speech_to_text
    gen_AI_text = summarize_text(speech_to_text)
    
    
    if gen_AI_text:
        # Step 2: 인식된 텍스트를 수어 동작으로 변환
        sign_videos = text_to_sign_language(gen_AI_text)
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
    
