from fastapi import  FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from pydantic import BaseModel
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
# Define request schema
class Translate(BaseModel):
    language: str
    text:str

# Route to handle translate
@app.post("/translate")
async def translate(req: Translate):
    try:
        response = chain.invoke({"language":req.language, "text": req.text})
        parsed = parser.parse(response)
        return {"success": True, "Translated_date": parsed}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port = 8000)


