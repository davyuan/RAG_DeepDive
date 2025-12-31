import os
import re
import chromadb
import dotenv

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from bs4 import BeautifulSoup

# --- CONFIGURATION ---
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
TEXT_LIMIT = 800
CHUNK_OVERLAP = 100

dotenv.load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_EP")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

def table_to_markdown(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
        rows.append(cells)

    md = []
    header = rows[0]
    md.append("<!--table-->")
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in rows[1:]:
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md)

def split_with_variable_sizes(raw_text, text_limit=TEXT_LIMIT, chunk_overlap=CHUNK_OVERLAP):
    # Regex to find everything between <table> and </table>
    # Flags=re.S ensures . matches newlines
    blocks = re.split(r'(<table>.*?</table>)', raw_text, flags=re.S)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_limit,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_chunks = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        if "<table>" in block:
            # --- Handle Table ---
            block = table_to_markdown(block)
            final_chunks.append(block)
        else:
            # --- Handle Standard Text ---
            final_chunks.extend(text_splitter.split_text(block))
            
    return final_chunks


# 1. Initialize Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="growing_vegetables_azure")

# 2. Setup Header Strategy
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 3. Process Files
id_counter = 0

for file_name in os.listdir(DATA_PATH):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(DATA_PATH, file_name)
        
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=AZURE_ENDPOINT, 
            api_key=AZURE_KEY, 
            file_path=file_path, 
            api_model="prebuilt-layout",
            analysis_features=["ocrHighResolution"]          
        )
        
        raw_docs = loader.load() 
        
        for doc in raw_docs:
            # Step A: Split by Markdown headers (keeps titles attached to their data)
            sections = markdown_splitter.split_text(doc.page_content)
            
            documents_to_upsert = []
            metadatas_to_upsert = []
            ids_to_upsert = []

            for section in sections:
                # Step B: Apply our custom variable size splitting to each section
                chunks = split_with_variable_sizes(
                    section.page_content, 
                    text_limit=TEXT_LIMIT, 
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                for chunk_content in chunks:
                    documents_to_upsert.append(chunk_content)
                    ids_to_upsert.append(f"ID_{id_counter}")
                    
                    # Merge header metadata with general metadata
                    meta = section.metadata.copy()
                    meta.update({
                        "source": file_name,
                        "is_table": "<!--table-->" in chunk_content,
                        "char_size": len(chunk_content)
                    })
                    metadatas_to_upsert.append(meta)
                    id_counter += 1

            # 4. Upsert to Chroma
            if documents_to_upsert:
                collection.upsert(
                    documents=documents_to_upsert,
                    metadatas=metadatas_to_upsert,
                    ids=ids_to_upsert
                )

print(f"Finished! Loaded {id_counter} chunks (Tables with no size limit, Text up to 800 chars).")