from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Configure API and model
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", 
    groq_api_key=groq_api_key,
    temperature=0.1  # Lower temperature for more consistent extraction results
)

# Define the output schema using Pydantic
class TicketInfo(BaseModel):
    emp_id: str = Field(description="Employee ID of the person raising the ticket")
    name: str = Field(description="Full name of the employee")
    date: str = Field(description="Date when the ticket is raised (YYYY-MM-DD format)")
    country: str = Field(description="Country where the employee is located")
    category: str = Field(description="Category of the ticket (e.g. IT Support, HR, Facilities)")
    description: str = Field(description="Detailed description of the issue")
    severity: str = Field(description="Severity level of the issue (Low, Medium, High, Critical)")

# Input schema
class ChatMessage(BaseModel):
    message: str = Field(description="User's message describing their issue")

# Ticket response
class TicketResponse(BaseModel):
    ticket_id: str
    extracted_info: TicketInfo
    created_at: str

# Create JSON parser
parser = JsonOutputParser(pydantic_object=TicketInfo)

# Create the prompt template with detailed instructions
system_template = """You are an intelligent system designed to extract structured information from user messages for a ticket raising system. 

Parse the user's message and extract the following information in JSON format:
1. Employee ID (emp_id)
2. Full name (name)
3. Date (in YYYY-MM-DD format, default to today if not specified)
4. Country (country)
5. Category of the issue (category)
6. Detailed description of the issue (description)
7. Severity level (severity: Low, Medium, High, Critical)

If any information is missing, make a reasonable inference based on the context. If you can't infer a value, use a placeholder and mark it for follow-up.

{format_instructions}
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{message}")
])

# Add format instructions
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Create the processing chain
extraction_chain = prompt | model | parser

# Initialize FastAPI app
app = FastAPI(
    title="Ticket Raising System API", 
    version="1.0",
    description="An API for extracting structured ticket information from chat messages"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory ticket storage
tickets_db = {}

@app.post("/extract-ticket-info", response_model=TicketResponse)
async def extract_ticket_info(chat_message: ChatMessage):
    """
    Extract structured ticket information from a chat message
    """
    try:
        # Process the message with the extraction chain
        extracted_info = extraction_chain.invoke({"message": chat_message.message})

        
        # Create timestamp
        created_at = datetime.now().isoformat()
        
        # Store the ticket in our simple database
        tickets_db[ticket_id] = {
            "extracted_info": extracted_info.dict() if hasattr(extracted_info, "dict") else extracted_info,
            "created_at": created_at
        }
        
        # Return the response
        return TicketResponse(
            ticket_id=ticket_id,
            extracted_info=extracted_info,
            created_at=created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ticket: {str(e)}")

# @app.get("/tickets/{ticket_id}")
# async def get_ticket(ticket_id: str):
#     """
#     Retrieve a ticket by ID
#     """
#     if ticket_id not in tickets_db:
#         raise HTTPException(status_code=404, detail="Ticket not found")
    
#     return {
#         "ticket_id": ticket_id,
#         **tickets_db[ticket_id]
#     }

# @app.get("/tickets")
# async def get_all_tickets():
#     """
#     Retrieve all tickets
#     """
#     return {
#         "tickets": [
#             {"ticket_id": tid, **data}
#             for tid, data in tickets_db.items()
#         ]
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)