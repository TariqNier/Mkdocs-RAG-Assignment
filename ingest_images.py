import os
import time
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# 1. SETUP
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # If .env is missing, ask user to paste it here for the script to run
    api_key = input("Paste your Google API Key: ")

genai.configure(api_key=api_key)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# Connect to the SAME database we used for text
print("🔌 Connecting to database...")
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path="db_new/") # Make sure this matches your folder!
collection = chroma_client.get_collection(name="mkdocs_rag", embedding_function=local_ef)

# 2. FIND IMAGES
docs_path = "mkdocs_repo/docs"
image_extensions = (".png", ".jpg", ".jpeg", ".gif")

print(f"📷 Scanning {docs_path} for images...")

count = 0
for root, dirs, files in os.walk(docs_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            file_path = os.path.join(root, file)
            
            try:
                # Open image
                img = Image.open(file_path)
                
                # Ask Gemini to describe it
                print(f"   - Analyzing {file}...", end="")
                response = vision_model.generate_content([
                    "Describe this technical documentation image in detail. Keywords only.", 
                    img
                ])
                description = response.text
                
                # Save to DB
                # We add a special metadata tag: type="image"
                collection.add(
                    documents=[description],
                    ids=[f"image-{file}"],
                    metadatas=[{
                        "source": file_path, 
                        "type": "image" 
                    }]
                )
                print(" ✅ Added caption.")
                count += 1
                
                # Sleep briefly to be nice to the free API
                time.sleep(1)
                
            except Exception as e:
                print(f" ❌ Failed: {e}")

print(f"🎉 Processed {count} images!")