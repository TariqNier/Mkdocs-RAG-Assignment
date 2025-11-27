# MkDocs RAG Assistant (Text + Multimodal)

## 📌 Project Overview
This project is a Retrieval-Augmented Generation (RAG) application that allows users to chat with the MkDocs open-source documentation. It supports both **Text** queries and **Multimodal (Image)** retrieval.

## 🛠️ Technical Implementation (Deliverables)

### 1. Chunking Method
* **Selected Method:** `MarkdownHeaderTextSplitter` (LangChain)
* **Reason:** MkDocs documentation is written in Markdown and structured via headers (`#`, `##`). Splitting by headers preserves the semantic context of sections (e.g., keeping all "Configuration" settings together) better than arbitrary character splitting.

### 2. Cleaning Method
* **Method:** Regex Pre-processing
* **Details:** Implemented a Python function to remove YAML Frontmatter (metadata between `---` tags) from the beginning of files. This prevents non-content metadata from polluting the semantic search.

### 3. Embedding Model
* **Selected Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Reason:** A high-performance, open-source local model. It avoids API latency/timeouts during large ingestion jobs and runs efficiently on standard hardware.

### 4. Vector Database
* **Selected DB:** `ChromaDB` (Persistent)
* **Reason:** Open-source and file-based. It requires no external server setup (unlike Pinecone/Weaviate) and allows saving the index to disk (`db_new/` folder) for reuse across sessions.

### 5. Application & Bonus Features
* **Frontend:** Built with **Streamlit** for a chat-based UI.
* **LLM:** **Google Gemini 2.0 Flash** for answer generation.
* **Bonus 1 (Multimodal):** Implemented an image ingestion pipeline that uses Gemini Vision to caption images. Users can ask "Show me X," and the system retrieves and displays the relevant image.
* **Bonus 2:** Fully interactive web application.

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Ingest Data (If DB is missing):**
    ```bash
    python ingest.py  # For Text
    python ingest_images.py # For Images
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
