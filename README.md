<div align="center">

# 📚 Conversational RAG Application

### *Intelligent Document Q&A with AI-Powered Retrieval*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-Framework-green)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and Groq LLM. Upload PDF documents and have intelligent, context-aware conversations about their content using cutting-edge AI technology.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Documentation](#-documentation)

---

</div>

## 🌟 Features

### Core Capabilities

- 📄 **Multi-Document Processing**: Upload and process multiple PDF files simultaneously with intelligent text extraction
- 🔍 **Semantic Search**: Advanced vector-based document retrieval using ChromaDB and OpenAI embeddings
- 💬 **Conversational AI**: Maintains chat history for contextual, multi-turn question answering
- 🎯 **Session Management**: Support for multiple conversation sessions with independent contexts
- 🎨 **Modern UI**: Clean, intuitive Streamlit interface with real-time feedback
- ⚡ **Fast Inference**: Powered by Groq's high-performance LLM infrastructure
- 🛡️ **Error Handling**: Robust error management with user-friendly feedback messages
- 🏗️ **Scalable Architecture**: Modular design with clear separation of concerns

### Advanced Features

- **Context-Aware Retrieval**: Reformulates questions based on conversation history
- **Source Attribution**: View the exact document chunks used to generate answers
- **Configurable Parameters**: Easily adjust chunk sizes, model settings, and response length
- **Temporary File Management**: Automatic cleanup of uploaded files for security
- **Logging System**: Comprehensive logging for debugging and monitoring



### Application Workflow

```
┌─────────────┐
│ Upload PDFs │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Text Extraction │
│   & Chunking     │
└──────┬───────────┘
       │
       ▼
┌─────────────────┐
│ Create Embeddings│
│  (OpenAI API)    │
└──────┬───────────┘
       │
       ▼
┌─────────────────┐
│  Store in       │
│  ChromaDB       │
└──────┬───────────┘
       │
       ▼
┌─────────────────┐
│  Ask Questions  │
└──────┬───────────┘
       │
       ▼
┌─────────────────┐
│ Retrieve Relevant│
│   Documents      │
└──────┬───────────┘
       │
       ▼
┌─────────────────┐
│  Generate Answer │
│   (Groq LLM)     │
└──────┬───────────┘
       │
       ▼
┌─────────────────┐
│ Display Response │
│  + Chat History  │
└──────────────────┘
```

### Example Use Cases

| Use Case | Example Question |
|----------|------------------|
| **Research Analysis** | "What are the main findings of this study?" |
| **Document Summary** | "Summarize the key points from section 3" |
| **Comparative Analysis** | "How does approach A differ from approach B?" |
| **Information Extraction** | "What does the document say about climate change?" |
| **Contextual Follow-up** | "Can you elaborate on the methodology?" |

## 🏗️ Architecture

The application follows a clean, modular architecture:

```
├── RAGConfig           # Configuration management
├── DocumentProcessor   # PDF processing and vectorization
└── ConversationalRAG   # Chat chain and history management
```

### Key Components

1. **Document Processing Pipeline**
   - PDF loading with PyPDFLoader
   - Text splitting with RecursiveCharacterTextSplitter
   - Vector embedding with OpenAI embeddings
   - Storage in ChromaDB vector database

2. **Conversational Chain**
   - History-aware retrieval for context understanding
   - Question reformulation for standalone queries
   - LLM-powered answer generation with Groq
   - Session-based chat history management

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** installed on your system
- **OpenAI API key** - [Get yours here](https://platform.openai.com/api-keys)
- **Groq API key** - [Sign up here](https://console.groq.com/keys)
- **Git** (optional, for cloning)

### Quick Installation

#### Option 1: Clone Repository

```bash
# Clone the repository
git clone <your-repository-url>
cd conversational-rag-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

#### Option 2: Manual Setup

1. **Download** the project files
2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment**:
   - Add your OpenAI API key to `.env`:
     ```env
     OPENAI_API_KEY=sk-your-actual-api-key-here
     ```

### Running the Application

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`




## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | Streamlit |
| **LLM** | Groq (openai/gpt-oss-20b) |
| **Embeddings** | OpenAI Embeddings |
| **Vector DB** | ChromaDB |
| **Orchestration** | LangChain |
| **Document Parsing** | PyPDF |

