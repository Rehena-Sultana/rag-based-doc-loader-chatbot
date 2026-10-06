# RAG-Based Document Loader Chatbot 💬 📚

An interactive Retrieval-Augmented Generation (RAG) conversational interface built with **Gradio**, **LlamaIndex**, and **Groq**. Upload PDF and DOCX documents and ask questions directly based on document content.

---

## 🌟 Features

- 📄 **Multi-Format Document Support**: Extract text seamlessly from `.pdf` (via `pdfplumber`) and `.docx` (via `python-docx`).
- ⚡ **High-Speed RAG Pipeline**: Uses `LlamaIndex` (`VectorStoreIndex`) with `condense_question` chat mode for context-aware follow-up queries.
- 🧠 **Free Local Embeddings**: Built-in HuggingFace embedding model (`BAAI/bge-small-en-v1.5`) running locally.
- 🚀 **Groq LLM Acceleration**: Powered by Groq's high-speed inference engine utilizing the `openai/gpt-oss-20b` LLM model.
- 🔒 **Privacy & User Isolation**: Stores saved chat histories locally in user-specific JSON logs hashed from SHA-256 API key signatures (`conversations_<user_id>.json`).
- 🗂️ **Comprehensive Conversation Management**:
  - Save current chat sessions.
  - View full conversation history logs.
  - Filter conversation history by specific dates (`YYYY-MM-DD`).
  - Delete specific conversations or reset all history logs.

---

## 🛠️ Project Structure

```text
├── app.py              # Main Gradio UI application and RAG pipeline engine
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
```

---

## 📋 Prerequisites

- **Python 3.9+**
- **Groq API Key**: Free API key available at [console.groq.com](https://console.groq.com/).

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rehena-Sultana/rag-based-doc-loader-chatbot.git
   cd "rag-based doc loader chatbot"
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

Launch the app by executing `app.py`:

```bash
python app.py
```

Once started, open your web browser and navigate to the local URL provided (default: `http://127.0.0.1:7860`).

---

## 📖 How to Use

1. **Enter API Key**: Paste your Groq API Key into the **Groq API Key** field.
2. **Upload Files**: Drag & drop or select `.pdf` / `.docx` files.
3. **Index Documents**: Click **Load Documents** and wait for the confirmation status.
4. **Start Chatting**: Ask questions in the chat text box. The chatbot will answer strictly based on the uploaded document contents.
5. **Manage History**: Use the sidebar options to **Save**, **Load**, or **Delete** prior conversations.

---

## 📦 Core Dependencies

- [Gradio](https://www.gradio.app/) - Web application interface
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLM applications & vector index
- [llama-index-llms-groq](https://pypi.org/project/llama-index-llms-groq/) - Groq LLM integration
- [llama-index-embeddings-huggingface](https://pypi.org/project/llama-index-embeddings-huggingface/) - HuggingFace local embedding integration
- [pdfplumber](https://github.org/jsvine/pdfplumber) - PDF text extraction
- [python-docx](https://python-docx.readthedocs.io/) - Word document parsing
