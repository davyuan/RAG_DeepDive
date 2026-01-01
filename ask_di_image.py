import base64
import mimetypes
from pathlib import Path

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

def encode_image_to_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    b64_payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64_payload}"


def describe_image(client: OpenAI, image_path: Path) -> str:
    prompt = (
        "You are inspecting a photograph regarding vegetable planting. "
        "Describe what you see with an emphasis on identifying keywords, plant parts, or conditions "
        "that could be useful for searching a document database about Florida planting. "
        "Keep it factual and focused on observable details."
    )

    data_url = encode_image_to_data_url(image_path)

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    text_fragments = []
    for output in response.output or []:
        for item in getattr(output, "content", []):
            item_type = getattr(item, "type", None)
            if item_type == "output_text":
                text_fragments.append(getattr(item, "text", ""))

    return "\n".join(text_fragments).strip()


client = OpenAI()

user_query = input("What do you want to know about growing vegetables?\n\n")
image_input = input("Optional image path (leave blank to skip): \n").strip()

image_description = ""
if image_input:
    image_path = Path(image_input)
    if image_path.exists():
        image_description = describe_image(client, image_path)
    else:
        print(f"Warning: {image_input} does not exist; skipping vision step.")

combined_query = "\n".join(filter(None, [user_query.strip(), image_description.strip()]))
if not combined_query:
    raise SystemExit("No query or image description provided.")

results = collection.query(
    query_texts=[combined_query],
    n_results=4
)

#print(results['documents'])
#print(results['metadatas'])

reranked_results = rerank_results(results)

system_prompt = generate_rag_sys_prompt(reranked_results)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                user_query
                if not image_description
                else f"{user_query}\n\nImage details:\n{image_description}"
            ),
        },
    ],
)

print("\n\n---------------------\n\n")

print(response.choices[0].message.content)