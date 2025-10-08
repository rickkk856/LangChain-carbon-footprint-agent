# 🌍 LangChain – Carbon Footprint API

This repository contains a **FastAPI application** powered by [LangChain](https://python.langchain.com/) for analyzing the **carbon footprint of architectural projects**.  

The agent integrates **Google Gemini 2.0 Flash** with LangChain's agent framework, memory management, and tool ecosystem to provide structured, actionable sustainability insights.  

The AI agent can:
- Extract project details (type, location, materials, energy systems, water/waste management, etc.)
- Estimate carbon footprint across lifecycle stages:
  - **Embodied carbon** (materials, transport, construction)  
  - **Operational carbon** (energy and water use during building life)  
  - **End-of-life carbon** (demolition, disposal, recycling potential)  
- Return structured outputs with:
  - Total estimated footprint (kgCO₂e / tCO₂e)  
  - Lifecycle breakdown  
  - Key emission drivers  
  - Suggested mitigation strategies (e.g., low-carbon concrete, passive design, renewables)  

The API exposes both **synchronous** and **streaming** endpoints for real-time interaction, making it suitable for integration into dashboards, design tools, or research workflows.

[▶️YouTube🎥 - Watch the demo](https://www.youtube.com/watch?v=jm0TCv11ZVQ)

[![▶️YouTube🎥 - Watch the demo](img.png)](https://www.youtube.com/watch?v=jm0TCv11ZVQ)

---

## 🏗️ Architecture

This application uses:
- **LangChain** for agent orchestration and tool calling
- **Google Gemini 2.0 Flash** via `langchain-google-genai` for LLM capabilities
- **File-based chat history** for persistent conversation memory
- **Sliding window memory** (20 messages) to manage context efficiently
- **HTTP request tools** for fetching and analyzing web content (project URLs)
- **FastAPI** for REST API endpoints with streaming support

---

## 🚀 Setup / Usage

### 1. Create Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
.venv\Scripts\Activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file with your API keys (see `.env_example` for template):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

[🔑 Generate Gemini API Keys Here](https://aistudio.google.com/app/api-keys)

### 5. Run the Application
```bash
python app.py
```

### 6. Access the Application
- **Web Interface:** Open `http://localhost:8000` to interact with the agent
- **API Documentation:** Open `http://localhost:8000/docs` to see the interactive API docs (Swagger UI)

---

## 🧪 Query Examples

### Non-Streaming Request
```bash
curl -X 'POST' \
  'http://localhost:8000/carbon' \
  -H 'accept: text/plain' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Analyze the carbon footprint of this project: https://www.archdaily.com.br/br/776950/casa-vila-matilde-terra-e-tuma-arquitetos",
  "user_id": "demo-user",
  "session_id": "chat-session"
}'
```

### Streaming Request
```bash
curl -X 'POST' \
  'http://localhost:8000/carbon-streaming' \
  -H 'accept: text/plain' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Analyze the carbon footprint of this project: https://www.archdaily.com.br/br/776950/casa-vila-matilde-terra-e-tuma-arquitetos",
  "user_id": "demo-user",
  "session_id": "chat-session"
}'
```

### Python Example
```python
import requests

url = "http://localhost:8000/carbon"
payload = {
    "prompt": "Analyze the carbon footprint of this project: https://www.archdaily.com.br/br/776950/casa-vila-matilde-terra-e-tuma-arquitetos",
    "user_id": "demo-user",
    "session_id": "chat-session"
}

response = requests.post(url, json=payload)
print(response.text)
```

---

## 💾 Session Management

LangChain stores conversation history using `FileChatMessageHistory`. Session files are organized as follows:

```
./sessions/
  └── {user_id}/
      └── {session_id}.json    # Chat history with all messages
```

### How It Works:
- Each user has a dedicated folder under `sessions/`
- Each session is stored as a separate JSON file
- The **sliding window memory** keeps only the most recent 20 messages in context
- Older messages are preserved in the file but not loaded into the agent's context

### Customization:
To adjust the sliding window size, modify the `SlidingWindowMemory` class initialization:

```python
sliding_window = SlidingWindowMemory(window_size=20)  # Change 20 to your desired size
```

For alternative memory strategies, check out:
- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [LangChain Chat Message Histories](https://python.langchain.com/docs/integrations/memory/)

---

## 🔧 Key Components

### Agent Tools
The agent has access to:
- **RequestsGetTool**: Fetch content from URLs via HTTP GET
- **RequestsPostTool**: Send data to URLs via HTTP POST
- **ready_to_summarize**: Custom tool to signal when summarization begins (streaming mode)

⚠️ **Security Note**: HTTP request tools are enabled with `allow_dangerous_requests=True`. Be cautious about which URLs you allow the agent to access. Consider implementing URL allowlisting for production use.

### Model Configuration
Currently using **Gemini 2.0 Flash Experimental** with:
- Temperature: 0.7
- Max tokens: 1000
- Streaming support enabled for real-time responses

To switch models, modify the `create_llm()` function:
```python
def create_llm(streaming: bool = False):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Change this
        google_api_key=GEMINI_API,
        temperature=0.7,
        max_tokens=1000,
        streaming=streaming
    )
```

Available Gemini models:
- `gemini-2.0-flash-exp` (experimental, fastest)
- `gemini-1.5-pro` (balanced)
- `gemini-1.5-flash` (fast and efficient)

---

## 🎯 System Prompt

The agent follows a comprehensive system prompt that guides it to:
1. Extract architectural project information from URLs
2. Estimate carbon footprint using recognized frameworks (IPCC, LEED, BREEAM, RICS)
3. Provide structured outputs with clear units and assumptions
4. Suggest mitigation strategies

You can customize the behavior by modifying `CARBON_SYSTEM_PROMPT` in `app.py`.

---

## 📦 Requirements

Key dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `langchain` - Agent orchestration
- `langchain-google-genai` - Google Gemini integration
- `langchain-community` - Community tools and utilities
- `python-dotenv` - Environment variable management

See `requirements.txt` for complete list with versions.

---

## 📚 Useful Resources

### LangChain Documentation
- [LangChain Official Docs](https://python.langchain.com/)
- [Agents](https://python.langchain.com/docs/modules/agents/)
- [Tools](https://python.langchain.com/docs/modules/tools/)
- [Memory](https://python.langchain.com/docs/modules/memory/)
- [Google Gemini Integration](https://python.langchain.com/docs/integrations/platforms/google/)

### Related Projects
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Community](https://github.com/langchain-ai/langchain/tree/master/libs/community)

### Security
- [LangChain Security Best Practices](https://python.langchain.com/docs/security)

