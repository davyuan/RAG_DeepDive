import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# setting the environment

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

def rerank_results(results):
    """Convert query responses to a reranked list of documents."""
    zipped_results = []
    for i in range(len(results["ids"][0])):
        zipped_results.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if "distances" in results else None,
        })

    reranked = sorted(
        zipped_results,
        key=lambda x: (
            x["metadata"].get("is_table", False),
            -x["distance"] if x["distance"] is not None else 0,
        ),
        reverse=True,
    )

    return reranked

def generate_rag_sys_prompt(reranked_results):
    context_blocks = []
    
    for res in reranked_results:
        metadata = res["metadata"]
        content = res["document"]
        source = metadata.get("source", "Unknown")
        header = metadata.get("Header 1", metadata.get("Header 2", "General Info"))
        
        if metadata.get("is_table"):
            # Format table with a clear label
            block = f"--- TABLE FROM {source} ({header}) ---\n{content}\n"
        else:
            # Format text blocks
            block = f"--- TEXT FROM {source} ({header}) ---\n{content}\n"
        
        context_blocks.append(block)
    
    # Combine all blocks into a single context string
    context_str = "\n".join(context_blocks)
    
    # The System Prompt: Instructions for the LLM
    prompt = f"""
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
### Guidelines:
1. **Prioritize Tables:** If the answer is in a table, extract the specific data accurately.
2. **Stay Grounded:** Only answer based on the provided data. If the answer isn't there, say you don't know.
3. **Formatting:** Use a friendly, helpful tone.
--------------------
The data:

{context_str}

"""
    return prompt


chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="growing_vegetables_azure")


user_query = input("What do you want to know about growing vegetables?\n\n")

results = collection.query(
    query_texts=[user_query],
    n_results=4
)

#print(results['documents'])
#print(results['metadatas'])

reranked_results = rerank_results(results)

client = OpenAI()

system_prompt = generate_rag_sys_prompt(reranked_results)

print(system_prompt)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_query}    
    ]
)

print("\n\n---------------------\n\n")

print(response.choices[0].message.content)