# LanguageConverter


### 음성을 수어로 변환하는 프로젝트

이 저장소는 음성을 수어 비디오로 변환하는 Python 스크립트를 포함하고 있습니다.
이 프로젝트는 음성 인식 및 자연어 처리를 활용하여 음성을 해석하고 해당 단어를 수어 비디오에 매핑하고 있습니다.

### 기능

- **음성 인식**: 음성을 텍스트로 변환합니다.
- **텍스트 토크나이징**: 인식된 텍스트를 단어로 분리합니다.
- **수어 매핑**: 각 단어를 해당하는 수어 비디오에 매핑합니다.
- **비디오 재생**: 인식된 단어에 해당하는 수어 비디오를 검색하고 재생합니다.

## 요구 사항

- Python 3.x
- nltk
- opencv-python
- speech_recognition

## 설치 방법

1. 저장소를 클론합니다:
    ```bash
    git clone https://github.com/yourusername/speech-to-sign-language-converter.git
    ```
2. 프로젝트 디렉토리로 이동합니다:
    ```bash
    cd speech-to-sign-language-converter
    ```
3. 필요한 Python 패키지를 설치합니다:
    ```bash
    pip install -r requirements.txt
    ```

### 사용 방법

### sign_language_converter.py

이 스크립트는 텍스트를 해당하는 수어 비디오로 변환합니다.

#### 예제:

```python
from sign_language_converter import text_to_sign_language

text = "병원"
videos = text_to_sign_language(text)
print(videos)  # 출력: ['hospital_sign.mp4']
