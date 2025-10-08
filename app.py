from collections.abc import Callable
from typing import Iterator, Dict, Optional, Any
from uuid import uuid4
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import os
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.tools import RequestsGetTool, RequestsPostTool
from langchain.tools import Tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
import json
from pathlib import Path
import asyncio
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

#%% --- Environment Variables
from dotenv import load_dotenv

load_dotenv(".env")
GEMINI_API = os.getenv("GEMINI_API_KEY")

#%% --- Base Path
BASE_PATH = Path(__file__).resolve().parent

#%% --- Session Manager (LangChain version)
def get_session_history(user_id: str, session_id: str):
    """Create a file-based chat history for persistence."""
    session_dir = BASE_PATH / "sessions" / user_id
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = session_dir / f"{session_id}.json"
    return FileChatMessageHistory(str(session_file))

#%% --- Sliding Window Memory Manager
class SlidingWindowMemory:
    """Implements sliding window to keep only recent messages."""
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
    
    def filter_messages(self, messages):
        """Keep only the last window_size messages, preserving system message."""
        if len(messages) <= self.window_size:
            return messages
        
        # Keep system message if it exists
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # Keep most recent messages
        recent_msgs = other_msgs[-(self.window_size - len(system_msgs)):]
        
        return system_msgs + recent_msgs

#%% --- System Prompt
CARBON_SYSTEM_PROMPT = """You are an AI Agent specialized in carbon footprint analysis of architectural projects.

Your main task is to analyze the contents of a given URL (which may include text, images, PDFs, BIM models, or other documents) describing an architectural project and then:

Extract relevant information such as:

Project type (residential, commercial, commercial, etc.)
Location (country, climate zone, urban/rural setting)
Building size (floor area, height, number of floors)
Construction materials and quantities (concrete, steel, wood, glass, insulation, finishes, etc.)
Energy systems (HVAC, lighting, renewable sources, fossil-fuel use)
Water and waste management systems
Transportation or mobility considerations (e.g., parking, bike storage, public transit proximity)

Estimate carbon footprint for each stage of the building lifecycle:

Embodied carbon (extraction, manufacturing, transport, and construction of materials)
Operational carbon (heating, cooling, electricity, water use, lighting, appliances over the building's lifespan)
End-of-life carbon (demolition, disposal, recycling potential)

Provide outputs in a structured format, including:

Total estimated carbon footprint (in kgCO₂e or tCO₂e)
Breakdown by lifecycle stage (embodied, operational, end-of-life)
Key drivers of emissions (e.g., high cement use, inefficient HVAC, lack of renewable energy)
Suggested alternatives or mitigation strategies (e.g., use of low-carbon concrete, more insulation, renewable energy integration, timber instead of steel, passive design strategies).

Communicate clearly, using:

Numerical estimates with clear units (kgCO₂e/m², total tCO₂e)
Tables or bullet points where appropriate
A short plain-language summary for non-experts

Constraints:

If the URL lacks sufficient data, state assumptions clearly and explain uncertainties.
Follow recognized frameworks such as IPCC guidelines, LEED, BREEAM, or RICS Whole Life Carbon Assessment whenever possible.
Be transparent about data sources, assumptions, and calculation methods.

Your goal is to provide a reliable, structured, and actionable carbon footprint analysis to help architects, engineers, and stakeholders make informed decisions about sustainability.
"""

class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    session_id: str

#%% --- Model Configuration
def create_llm(streaming: bool = False):
    """Create LangChain Gemini model."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GEMINI_API,
        temperature=0.7,
        max_tokens=1000,
        streaming=streaming
    )

#%% --- Tools Setup
# Create a wrapper that handles HTTP requests safely
#requests_wrapper = UnsafeRequestsWrapper()


# HTTP request tools for fetching URLs.
# Requests can be dangerous and can lead to security vulnerabilities. 
# For example, users can ask a server to make a request to an internal server. 
# It's recommended to use requests through a proxy server and avoid accepting inputs from untrusted sources without proper sandboxing.
# Please see: https://python.langchain.com/docs/security for further security information.
#requests_get = RequestsGetTool(requests_wrapper=requests_wrapper)
#requests_post = RequestsPostTool(requests_wrapper=requests_wrapper)

requests_wrapper = TextRequestsWrapper(headers={})

# Then create tools with both wrapper and allow_dangerous_requests
requests_get = RequestsGetTool(
    requests_wrapper=requests_wrapper,
    allow_dangerous_requests=True
)
requests_post = RequestsPostTool(
    requests_wrapper=requests_wrapper,
    allow_dangerous_requests=True
)


def create_ready_to_summarize_tool():
    """Tool to signal when agent is ready to summarize."""
    def ready_to_summarize_func(input: str = "") -> str:
        """Signal that the agent is ready to provide the summary."""
        return "Ok - continue providing the summary!"
    
    return Tool(
        name="ready_to_summarize",
        description="Call this tool right before you summarize the carbon footprint analysis response.",
        func=ready_to_summarize_func
    )

# Available tools
base_tools = [requests_get, requests_post]
streaming_tools = base_tools + [create_ready_to_summarize_tool()]

#%% --- Agent Creation
def create_carbon_agent(user_id: str, session_id: str, streaming: bool = False):
    """Create a LangChain agent with memory."""
    llm = create_llm(streaming=streaming)
    tools = streaming_tools if streaming else base_tools
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", CARBON_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    # Add memory with sliding window
    sliding_window = SlidingWindowMemory(window_size=20)
    
    def get_filtered_history(session_id_key: str):
        history = get_session_history(user_id, session_id)
        messages = history.messages
        filtered = sliding_window.filter_messages(messages)
        # Update history with filtered messages
        history.clear()
        for msg in filtered:
            history.add_message(msg)
        return history
    
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: get_filtered_history(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return agent_with_memory

#%% --- FastAPI App Setup
app = FastAPI(title="Carbon Footprint API")

# CORS middleware
origins = [
    "*",
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#%% --- Serve index.html
app.mount("/static", StaticFiles(directory="."), name="static") 

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

#%% --- API Endpoints
@app.get('/health')
def health_check():
    """Health check endpoint for the load balancer."""
    return {"status": "healthy"}

@app.post('/carbon')
async def get_carbon(request: PromptRequest):
    """Endpoint to get carbon footprint information (non-streaming)."""
    prompt = request.prompt
    
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    try:
        agent = create_carbon_agent(request.user_id, request.session_id, streaming=False)
        
        response = await agent.ainvoke(
            {"input": prompt},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        content = response.get("output", str(response))
        return PlainTextResponse(content=content)
    except Exception as e:
        return PlainTextResponse(content=f"Error: {str(e)}", status_code=500)


async def stream_agent_response(prompt: str, request: PromptRequest):
    """
    Stream agent responses as they are generated.
    
    Yields text chunks and tool usage notifications in real-time.
    """
    try:
        # Create agent for this request
        agent = create_carbon_agent(
            user_id=request.user_id, 
            session_id=request.session_id,
            streaming=True
        )
        
        # Get session history
        history = get_session_history(request.user_id, request.session_id)
        sliding_window = SlidingWindowMemory(window_size=20)
        filtered_messages = sliding_window.filter_messages(history.messages)
        
        # Stream agent responses using astream_events
        full_response = ""
        async for event in agent.astream_events(
            {"input": prompt},
            config={"configurable": {"session_id": request.session_id}},
            version="v1"
        ):
            kind = event["event"]
            
            # Handle tool calls
            if kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                yield f"\n\n🔧 Using tool: {tool_name}\n"
            
            # Handle LLM token streaming
            elif kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Check for ready_to_summarize signal
                    if "ready_to_summarize" in str(content).lower():
                        yield "\n"
                    else:
                        full_response += content
                        yield content
        
        # Save conversation to history after streaming completes
        if full_response:
            history.add_user_message(prompt)
            history.add_ai_message(full_response)
                
    except Exception as e:
        error_msg = f"\n\nError: {str(e)}"
        print(f"Error streaming carbon agent: {e}")  # Log error
        yield error_msg

@app.post('/carbon-streaming')
async def get_carbon_streaming(request: PromptRequest):
    """Endpoint to stream the carbon footprint summary as it comes in."""
    try:
        prompt = request.prompt

        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")

        return StreamingResponse(
            stream_agent_response(prompt, request),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # Get port from environment variable or default to 8000
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)