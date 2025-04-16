from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#KAG
qwen = ChatOllama(model="qwen2.5:3b", stop=["</s>"])