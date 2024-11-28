# ProjectAssistant
유튜브 기반 아이디어 생성프로그램
OpenAI API를 활용하여 진행했습니다.

## Conda 가상환경 실행 방법

1. `environment.yml` 파일을 사용하여 가상환경을 생성합니다:
    ```bash
    conda env create -f environment.yml
    ```

2. 가상환경을 활성화합니다:
    ```bash
    conda activate [환경이름]
    ```

3. 가상환경을 비활성화합니다:
    ```bash
    conda deactivate
    ```

## .env 파일     
    OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
   
## streamlit run
    ```bash
    streamlit run YTPI_UI
    ```    
#### 주의사항
- `.env` 파일은 **절대로** Git에 커밋하지 마세요. 민감한 정보를 보호하기 위해 `.gitignore` 파일에 `.env`를 추가하는 것을 잊지 마세요.
- 필요한 값들을 적절히 수정하여 환경에 맞게 설정하세요.
