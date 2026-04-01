# GenAI RAG Demo (Gemini + Chroma)

This project demonstrates a simple Retrieval-Augmented Generation (RAG) system using modern LangChain components. It connects to a local Chroma vector database, retrieves relevant context, and uses Google's Gemini LLM to generate an answer to a query.

## Overview

The project provides two primary ways to interact with the system:
1. **Web UI (`api.py`):** A FastAPI-based server providing an interactive browser UI.
2. **CLI Demo (`demo.py`):** A demonstration script for query execution and file output.

The RAG application performs the following actions:
1. **Environment Setup:** Loads environment variables (API keys) from a `.env` file.
2. **Vector Store Loading:** Initializes a Chroma vector database from the local `./chroma_knowledge_base` directory using `SentenceTransformerEmbeddings` (`all-MiniLM-L6-v2` model).
3. **LLM Initialization:** Configures the `gemini-2.5-flash` model via `ChatGoogleGenerativeAI`.
4. **Retrieval Chain setup:** Sets up a `RetrievalQA` chain using Maximum Marginal Relevance (MMR) search to fetch relevant context from the database.
5. **Query Execution:** Generates an answer along with the source documents used as context, providing them to either the web user interface or the CLI (`rag_output.txt`).

## Prerequisites

- **Python:** Ensure you have Python installed.
- **Environment Variables:** You must have a `.env` file in the root directory containing your Google API key:
  ```env
  GOOGLE_API_KEY=your_actual_api_key_here
  ```
- **Vector Database:** The script expects a pre-populated Chroma vector database in the `./chroma_knowledge_base` folder.

## Setup and Installation

1. Clone or navigate to the project repository.
2. Create and activate a virtual environment (recommended).
3. Install the required dependencies:
   ```bash
   pip install python-dotenv langchain-community langchain-google-genai sentence-transformers chromadb fastapi uvicorn pydantic
   ```
4. Make sure your `.env` file is set up and the `chroma_knowledge_base` directory exists with valid data.

## Usage

### Web Interface (Recommended)

To launch the interactive chat interface, run the FastAPI server:

```bash
uvicorn GenAI_project.api:app --reload
```

Then, open your web browser and navigate to `http://127.0.0.1:8000` to interact with the AI assistant.

### CLI Demo

To run the demonstration script, navigate to the `GenAI_project` directory or run:

```bash
python demo.py
```

The console will print progress messages as it loads the components. Once completed, it will output a new file named `rag_output.txt` containing the LLM's answer and the specific document snippets it retrieved to form that answer.
