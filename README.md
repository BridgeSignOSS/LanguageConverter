# Speech-to-Sign Language Translator


### 음성을 수어로 변환하는 프로젝트

이 저장소는 음성을 수어 비디오로 변환하는 Python 스크립트를 포함하고 있습니다.
이 프로젝트는 음성 인식 및 자연어 처리를 활용하여 음성을 해석하고 해당 단어를 수어 비디오에 매핑하고 있습니다.

### 기능

- **음성 인식**: 음성을 텍스트로 변환합니다.
- **텍스트 토크나이징**: 인식된 텍스트를 단어로 분리합니다.
- **수어 매핑**: 각 단어를 해당하는 수어 비디오에 매핑합니다.
- **비디오 재생**: 인식된 단어에 해당하는 수어 비디오를 검색하고 재생합니다.

## 사용된 기술

Speech Recognition: 음성 인식을 통해 마이크 입력을 텍스트로 변환
Ollama: 커스텀 Llama3 모델을 사용하여 텍스트를 재구성
NLTK: 텍스트를 토큰화
OpenCV: 수어 영상 파일 재생

## Llama3 모델 커스터마이징
이 프로젝트에서 사용된 Llama3 모델은 Ollama를 사용하여 커스터마이징되었습니다.

세부 내용:
모델: bridgesign
Temperature: 0으로 설정하여 모델이 일관된 응답을 생성하도록 함
System Prompt: 모델에게 의료 절차와 지시와 관련된 특정 문구를 식별하고 재구성하도록 설계

## Ollama를 사용한 모델 커스터마이징 예제
Ollama 라이브러리의 모델은 프롬프트를 사용하여 커스터마이징할 수 있습니다.

예시)
1. 모델 가져오기:
    ```bash
    ollama pull llama3
    ```
2. Modelfile 생성:
    ```bash
    FROM llama3
    # set the temperature to 0 [higher is more creative, lower is more coherent]
    PARAMETER temperature 0
    # set the system message
    SYSTEM """
    You are a helpful assistant designed to aid in medical settings by identifying and rephrasing sentences to include only specific phrases related to medical procedures and instructions, presented in lowercase and separated by commas.
    Here is a list of phrases you should look for in the input text: [where does it hurt, no fever, high fever, name and birthdate, administer anesthesia, .....].
    Please rephrase the given sentence from the doctor to include all relevant phrases from the list in lowercase, separated by commas. If no relevant phrases are found, just return the original sentence.
    """

    ```
3. 모델 생성 및 실행:
    ```bash
    ollama create bridgesign -f ./Modelfile
    ollama run bridgesign
    ```

## 사전 준비

1. 필요한 라이브러리 설치:
    ```bash
    pip install speechrecognition opencv-python nltk langchain_community
    ```
2.  NLTK 데이터 다운로드:
    ```bash
    import nltk
    nltk.download('punkt')
    ```
3. 동영상 파일 준비:
수어 영상 파일을 Your_video_folder 폴더에 저장합니다.
각 파일 이름은 sign_language_map 딕셔너리의 키 값과 일치해야 합니다.

### 사용 방법

### sign_language_interpreter.py

이 스크립트는 사용자의 음성을 인식하고 이를 수어로 번역하여 재생하는 시스템입니다.

1. 프로젝트를 클론하거나 다운로드합니다.
2. 위의 코드와 설명에 따라 환경을 설정합니다.
3. 터미널에서 코드를 실행합니다:
```bash
python script_name.py
```
4. 마이크를 통해 음성을 입력합니다. 예를 들어, "Let me check your temperature"라고 말합니다.
5. 입력된 음성은 텍스트로 변환되고, Ollama 모델을 통해 재구성됩니다.
6. 재구성된 텍스트는 미리 정의된 수어 영상 파일과 매핑됩니다.
7. 매핑된 수어 영상 파일이 OpenCV를 사용하여 재생됩니다.
