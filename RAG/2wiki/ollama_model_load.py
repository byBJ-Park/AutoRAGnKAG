# ollama_model_load.py (수정된 코드)
from langchain_ollama import ChatOllama

deepseek_r1 = ChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:01434",  # 01434 포트 지정
    stop=["</s>"]
)