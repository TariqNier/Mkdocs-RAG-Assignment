import os
import shutil
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. SETUP DATABASE
print("🔌 Setting up database in 'db_new'...")
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Force creation in the current folder
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "db_new")

if os.path.exists(db_path):
    shutil.rmtree(db_path)

chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.create_collection(name="mkdocs_rag", embedding_function=local_ef)

# 2. DOWNLOAD DATA
if not os.path.exists("mkdocs_repo"):
    # Using os.system for git since we are in a script, not a notebook
    os.system("git clone https://github.com/mkdocs/mkdocs.git mkdocs_repo")

docs_path = os.path.join("mkdocs_repo", "docs")
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2")])

# 3. INGEST
print("⏳ Processing files...")
count = 0
for root, dirs, files in os.walk(docs_path):
    for file in files:
        if file.endswith(".md"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                content = re.sub(r"^---\n(.*?)\n---\n", "", f.read(), flags=re.DOTALL)
            
            chunks = markdown_splitter.split_text(content)
            if chunks:
                collection.add(
                    documents=[c.page_content for c in chunks],
                    ids=[f"{file}-{i}" for i in range(len(chunks))],
                    metadatas=[{"source": file} for _ in chunks]
                )
                count += 1
print(f"✅ DONE! Added {count} files to: {db_path}")