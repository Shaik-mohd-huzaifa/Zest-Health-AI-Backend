import os
from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

LLM = ChatOpenAI(openai_api_base=base_url, openai_api_key=api_key, model_name="aria")

# print(LLM.invoke("What is the Capital of Spain"))
