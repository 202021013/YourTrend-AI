import os
import subprocess
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# .env 파일 로드
load_dotenv()

# OpenAI API 키 가져오기
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    print("Error: OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

# OpenAI 객체 초기화
llm = OpenAI(temperature=0.7, api_key=openai_api_key)

# 예 / 아니요 질문 후 .conda 환경 설정
def setup_conda_environment(env_name):
    try:
        subprocess.check_call(["conda", "create", "-n", env_name, "python=3.11", "-y"])
        print(f"Conda 환경 '{env_name}'이 성공적으로 생성되었습니다.")
    except subprocess.CalledProcessError:
        print(f"Conda 환경 '{env_name}' 생성 중 오류가 발생했습니다.")

# 주어진 주제 코드 생성
def generate_code(topic, language):
    prompts = {
        "en-US": "Write a Python script related to the following topic: {topic}. Include necessary imports and comments.",
        "ko-KR": "다음 주제와 관련된 파이썬 스크립트를 작성하세요: {topic}. 필요한 import문과 주석을 포함하세요."
    }
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=prompts[language],
    )
    chain = (
        {"topic": lambda x: x} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return chain.invoke(topic)

# 사용자의 코드 검토 요청
def review_code(code, language):
    prompts = {
        "en-US": "Review the following Python code and provide feedback:\n\n{code}\n\nFeedback:",
        "ko-KR": "다음 파이썬 코드를 검토하고 피드백을 제공하세요:\n\n{code}\n\n피드백:"
    }
    prompt = PromptTemplate(
        input_variables=["code"],
        template=prompts[language],
    )
    chain = (
        {"code": lambda x: x} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return chain.invoke(code)

def create_vector_store(code):
    """
    코드로부터 벡터 저장소를 생성합니다.
    """
    with open("temp_code.py", "w") as f:
        f.write(code)
    
    loader = TextLoader("temp_code.py")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    return FAISS.from_documents(texts, embeddings)

# 코드에 대한 질문에 답변
def answer_questions(vectorstore, question, language):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.3, api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa.invoke(question)

def main():
    print("AI 코드 생성기에 오신 것을 환영합니다!")

    # 언어 선택
    language = input("언어를 선택하세요 (en-US 또는 ko-KR): ").strip()
    if language not in ["en-US", "ko-KR"]:
        print("지원되지 않는 언어입니다. en-US로 설정합니다.")
        language = "en-US"

    # Conda 환경 설정 (선택적)
    setup_conda = input("Conda 환경을 설정하시겠습니까? (y/n): ").lower() == 'y'
    if setup_conda:
        env_name = input("Conda 환경 이름을 입력하세요: ")
        setup_conda_environment(env_name)

    # 주제 입력 받기
    topic = input("코드 생성을 위한 주제를 입력하세요: ")

    if topic:
        # 코드 생성
        generated_code = generate_code(topic, language)
        print("\n생성된 코드:")
        print(generated_code)
        
        # 코드 검토
        review = review_code(generated_code, language)
        print("\n코드 리뷰:")
        print(review)
        
        # 벡터 저장소 생성
        vectorstore = create_vector_store(generated_code)
        
        # 코드에 대한 질문 답변
        while True:
            question = input("\n코드에 대해 질문하세요 (종료하려면 'q' 입력): ")
            if question.lower() == 'q':
                break
            answer = answer_questions(vectorstore, question, language)
            print("답변:", answer)
        
        # 임시 파일 삭제
        os.remove("temp_code.py")
    else:
        print("주제를 입력해주세요.")

if __name__ == "__main__":
    main()