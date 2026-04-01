from dotenv import load_dotenv
load_dotenv(dotenv_path="GenAI_project/.env")

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from GenAI_project.rag_query import run_rag_query_ui

app = FastAPI(title="Cancer RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
def get_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cancer RAG Assistant</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
            .chat-container { width: 100%; max-width: 800px; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); display: flex; flex-direction: column; overflow: hidden; height: 90vh; }
            .header { background: #2c3e50; color: white; padding: 20px; text-align: center; font-size: 1.5rem; font-weight: bold; }
            .messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 80%; padding: 12px 16px; border-radius: 8px; line-height: 1.5; word-wrap: break-word; }
            .user-msg { background: #3498db; color: white; align-self: flex-end; border-bottom-right-radius: 0; }
            .bot-msg { background: #ecf0f1; color: #2c3e50; align-self: flex-start; border-bottom-left-radius: 0; }
            .sources { font-size: 0.85em; margin-top: 10px; padding-top: 10px; border-top: 1px solid #bdc3c7; color: #7f8c8d; }
            .input-area { display: flex; padding: 15px; background: #ecf0f1; border-top: 1px solid #ddd; }
            input[type="text"] { flex: 1; padding: 12px; border: 1px solid #bdc3c7; border-radius: 6px; outline: none; font-size: 1rem; }
            button { background: #2c3e50; color: white; border: none; padding: 12px 20px; margin-left: 10px; border-radius: 6px; cursor: pointer; font-size: 1rem; transition: background 0.3s; }
            button:hover { background: #1a252f; }
            .loading { align-self: flex-start; color: #7f8c8d; font-style: italic; display: none; margin-bottom: 10px;}
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="header">Cancer RAG Assistant</div>
            <div class="messages" id="chat-box">
                <div class="message bot-msg">Hello! I am your AI assistant specialized in medical information based on our documents. What would you like to know?</div>
                <div class="loading" id="loading">Thinking...</div>
            </div>
            <div class="input-area">
                <input type="text" id="query-input" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
                <button onclick="askQuestion()">Ask</button>
            </div>
        </div>
        <script>
            async function askQuestion() {
                const input = document.getElementById("query-input");
                const query = input.value.trim();
                if (!query) return;

                const chatBox = document.getElementById("chat-box");
                const loading = document.getElementById("loading");

                // Add User message
                const userDiv = document.createElement("div");
                userDiv.className = "message user-msg";
                userDiv.textContent = query;
                chatBox.insertBefore(userDiv, loading);
                
                input.value = "";
                loading.style.display = "block";
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query: query })
                    });
                    const data = await response.json();

                    loading.style.display = "none";

                    // Add Bot message
                    const botDiv = document.createElement("div");
                    botDiv.className = "message bot-msg";
                    botDiv.innerHTML = data.answer.replace(/\\n/g, '<br>');
                    
                    if (data.sources && data.sources.length > 0) {
                        let sourcesHtml = "<div class='sources'><strong>Sources:</strong><ul>";
                        data.sources.forEach(source => {
                            let sourceName = source.metadata ? source.metadata.source : 'Unknown File';
                            if (sourceName.includes('/')) sourceName = sourceName.split('/').pop();
                            if (sourceName.includes('\\\\')) sourceName = sourceName.split('\\\\').pop();
                            sourcesHtml += `<li>${sourceName}</li>`;
                        });
                        sourcesHtml += "</ul></div>";
                        botDiv.innerHTML += sourcesHtml;
                    }

                    chatBox.insertBefore(botDiv, loading);
                    chatBox.scrollTop = chatBox.scrollHeight;
                } catch (error) {
                    loading.style.display = "none";
                    const errDiv = document.createElement("div");
                    errDiv.className = "message bot-msg";
                    errDiv.style.color = "red";
                    errDiv.textContent = "Error: Could not connect to the server.";
                    chatBox.insertBefore(errDiv, loading);
                }
            }

            function handleKeyPress(e) {
                if (e.key === "Enter") askQuestion();
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/ask")
def ask_question(req: QueryRequest):
    answer, sources = run_rag_query_ui(req.query)
    return {
        "question": req.query,
        "answer": answer,
        "sources": sources
    }
