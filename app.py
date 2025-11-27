import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="MkDocs RAG Bot", page_icon="📘")


api_key = os.environ.get("GOOGLE_API_KEY") 
if not api_key:
    # If not in env, ask user for it
    with st.sidebar:
        api_key = st.text_input("Enter Google API Key", type="password")

if not api_key:
    st.warning("⚠️ Please enter your Google API Key to continue.")
    st.stop()

# 2. Configure Google Gemini
genai.configure(api_key=api_key)

# 3. Connect to Local Vector DB
# Note: We use the same 'all-MiniLM-L6-v2' model we used for ingestion
try:
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    chroma_client = chromadb.PersistentClient(path="db_new/") 
    collection = chroma_client.get_collection(
        name="mkdocs_rag",
        embedding_function=local_ef
    )
except Exception as e:
    st.error(f"Error connecting to Database: {e}")
    st.stop()

# --- CHAT UI ---
st.title("📘 MkDocs Assistant (Multimodal)")
question = st.chat_input("Ask about MkDocs...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results = collection.query(query_texts=[question], n_results=15)
            
            # 1. Handle Text vs Images
            context_text = []
            images_found = []
            
            if results['documents'][0]:
                for idx, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][idx]
                    
                    # If it's an image, save it to show later
                    if meta.get("type") == "image":
                        images_found.append(meta["source"])
                        # We also add the description to context so Gemini knows what the image is
                        context_text.append(f"[Image: {doc}]")
                    else:
                        # It's normal text
                        context_text.append(doc)
                        
                # 2. Generate Answer
                full_context = "\n\n".join(context_text)
                
                prompt = f"""
                Answer the question based on the context.
                If the context mentions an image, refer to it.
                
                Context:
                {full_context}
                
                Question: {question}
                """
                
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                
                st.markdown(response.text)
                
                # 3. Show Images (The Bonus Feature!)
                if images_found:
                    st.write("---")
                    st.caption("📷 Relevant Images found in docs:")
                    cols = st.columns(len(images_found))
                    for i, img_path in enumerate(images_found):
                        # Verify file exists before trying to open
                        if os.path.exists(img_path):
                            st.image(img_path, caption=os.path.basename(img_path), width=300)
            else:
                st.warning("No info found.")