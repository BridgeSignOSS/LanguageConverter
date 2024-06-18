import requests
import cv2
import numpy as np

# Ollama 서버의 URL 설정 (Ollama 서버의 실제 URL로 대체하세요)
OLLAMA_SERVER_URL = 'https://1750-34-143-162-16.ngrok-free.app'

# 텍스트를 Ollama 서버에 전송하여 한국수어 이미지를 생성하는 함수
def generate_ksl_image(text):
    try:
        response = requests.post(
            f'{OLLAMA_SERVER_URL}/generate',
            json={'prompt': f"Korean Sign Language for '{text}'"}
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            image_url = response.json().get('image_url')
            return image_url
        else:
            print(f"Failed to generate image. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request to Ollama server failed: {e}")
        return None

# 한국수어 이미지를 다운로드하고 화면에 출력하는 함수
def display_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # OpenCV 윈도우 생성 및 이미지 출력
            cv2.namedWindow('KSL Image', cv2.WINDOW_NORMAL)
            while True:
                cv2.imshow('KSL Image', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            print(f"Failed to retrieve the image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request to image URL failed: {e}")

def main():
    text = input("Enter a phrase to translate to Korean Sign Language: ")

    image_url = generate_ksl_image(text)
    if image_url:
        display_image_from_url(image_url)

if __name__ == "__main__":
    main()
