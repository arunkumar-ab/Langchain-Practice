from fastapi import  FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema.runnable import RunnableMap

from langserve import add_routes
import os 
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=groq_api_key)

#1. Create prompt template

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])
parser = StrOutputParser()

# Create Chain
chain = prompt_template|model|parser

#App definition
app = FastAPI(title="Langchain Server", version="1.0",
              description="A simple API server using langchain runnable interface")

#Adding Chain routes
add_routes(
    app,
    chain,
    path= "/chain"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port = 8000)


