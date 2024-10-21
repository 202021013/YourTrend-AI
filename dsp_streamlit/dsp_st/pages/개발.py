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
import streamlit as st

# .env 파일 로드
load_dotenv()

# OpenAI API 키 가져오기
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    st.error("Error: OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

# OpenAI 객체 초기화
llm = OpenAI(temperature=0.7, api_key=openai_api_key)

def setup_conda_environment(env_name):
    try:
        subprocess.check_call(["conda", "create", "-n", env_name, "python=3.11", "-y"])
        st.success(f"Conda 환경 '{env_name}'이 성공적으로 생성되었습니다.")
    except subprocess.CalledProcessError:
        st.error(f"Conda 환경 '{env_name}' 생성 중 오류가 발생했습니다.")

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

def answer_questions(vectorstore, question, language):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.3, api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa.invoke(question)

def main():
    st.title("AI 코드 생성기")

    # 언어 선택
    language = st.selectbox("언어를 선택하세요", ["en-US", "ko-KR"], index=1)

    # Conda 환경 설정 (선택적)
    if st.checkbox("Conda 환경을 설정하시겠습니까?"):
        env_name = st.text_input("Conda 환경 이름을 입력하세요")
        if st.button("Conda 환경 생성"):
            setup_conda_environment(env_name)

    # 주제 입력 받기
    topic = st.text_input("코드 생성을 위한 주제를 입력하세요")

    if st.button("코드 생성") and topic:
        # 코드 생성
        generated_code = generate_code(topic, language)
        st.subheader("생성된 코드:")
        st.code(generated_code, language='python')

        # 코드 검토
        review = review_code(generated_code, language)
        st.subheader("코드 리뷰:")
        st.text(review)

        # 벡터 저장소 생성
        vectorstore = create_vector_store(generated_code)

        # 코드에 대한 질문 답변
        question = st.text_input("코드에 대해 질문하세요 (종료하려면 'q' 입력)")
        if question and question.lower() != 'q':
            answer = answer_questions(vectorstore, question, language)
            st.subheader("답변:")
            st.text(answer)

        # 임시 파일 삭제
        if os.path.exists("temp_code.py"):
            os.remove("temp_code.py")

if __name__ == "__main__":
    main()