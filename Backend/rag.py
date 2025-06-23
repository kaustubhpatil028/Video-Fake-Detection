import os
import google.generativeai as genai
import faiss
import numpy as np
import wikipedia
import textwrap

# --- Configuration (Hardcoded API Key - Use .env for production!) ---
# Replace "YOUR_GEMINI_API_KEY" with your actual Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyBVbQcYILiH1-JUH120u9z48_4ZHWrgftE" 

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "AIzaSyCWe2e7NNkBnOd1xTu5sIg3w-EZ0WkojhY":
    raise ValueError("Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key in the code.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Embedding Model Setup ---
EMBEDDING_MODEL = "models/embedding-001" # A good general-purpose embedding model
llm_model = genai.GenerativeModel('gemini-pro') # For text generation

def get_embedding(text):
    """Generates an embedding for the given text."""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# --- Text Chunking Function ---
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Splits text into chunks with overlap."""
    if not text:
        return []
    chunks = []
    current_chunk = ""
    words = text.split()
    for word in words:
        if len(current_chunk) + len(word) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            # Create overlap: approx 1/5th of the current chunk's words
            overlap_words = current_chunk.split()[-int(chunk_overlap/5):] 
            current_chunk = " ".join(overlap_words) + " " + word
        else:
            current_chunk += " " + word
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if chunk] # Remove empty chunks

# --- FAISS Database Setup ---
class FAISSDocumentStore:
    def __init__(self, embedding_dimension):
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.documents = [] # Stores original text chunks
        self.metadata = [] # Stores metadata like source URL

    def add_documents(self, texts, sources=None):
        """Adds documents to the FAISS index."""
        new_embeddings = []
        new_documents = []
        new_metadata = []

        for i, text in enumerate(texts):
            embedding = get_embedding(text)
            if embedding is not None:
                new_embeddings.append(embedding)
                new_documents.append(text)
                new_metadata.append({"source": sources[i] if sources and i < len(sources) else "unknown"})

        if new_embeddings:
            embeddings_np = np.array(new_embeddings).astype('float32')
            self.index.add(embeddings_np)
            self.documents.extend(new_documents)
            self.metadata.extend(new_metadata)
            print(f"Added {len(new_embeddings)} documents to FAISS index.")
        else:
            print("No new embeddings to add.")

    def search(self, query_text, k=3):
        """Searches the FAISS index for relevant documents."""
        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            return []

        query_embedding_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding_np, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents): # Ensure index is valid
                results.append({
                    "text": self.documents[idx],
                    "source": self.metadata[idx].get("source", "unknown"),
                    "distance": distances[0][i]
                })
        return results

# --- RAG Fact-Checking Logic with Wikipedia Only ---

def fact_check_wikipedia_only(claim, num_wiki_sentences=5, num_retrieved_chunks=3):
    """
    Performs fact-checking using RAG with Wikipedia integration only.
    Returns a clear verdict (FAKE or REAL) and a 2-3 line summary, suitable for display in an extension.
    """
    print(f"\n--- Fact-Checking: '{claim}' (Wikipedia Only) ---")
    document_store = FAISSDocumentStore(embedding_dimension=768) 
    # 1. Gather information from Wikipedia
    print(f"Searching Wikipedia for: {claim}")
    wiki_results = []
    try:
        page = wikipedia.page(claim, auto_suggest=True, redirect=True)
        summary = wikipedia.summary(claim, sentences=num_wiki_sentences, auto_suggest=True, redirect=True)
        wiki_results.append({"title": page.title, "snippet": summary, "url": page.url})
    except wikipedia.exceptions.PageError:
        print(f"No Wikipedia page found for: {claim}")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for '{claim}'. Options: {e.options}. Trying first option.")
        try:
            page = wikipedia.page(e.options[0], auto_suggest=True, redirect=True)
            summary = wikipedia.summary(e.options[0], sentences=num_wiki_sentences, auto_suggest=True, redirect=True)
            wiki_results.append({"title": page.title, "snippet": summary, "url": page.url})
        except Exception as inner_e:
            print(f"Could not get Wikipedia page even after disambiguation: {inner_e}")
    except Exception as e:
        print(f"Error searching Wikipedia: {e}")
    for res in wiki_results:
        chunks = chunk_text(res['snippet'])
        document_store.add_documents(chunks, sources=[res['url']] * len(chunks))
    if document_store.index.ntotal == 0:
        return {
            "response": "RAG Verdict: Unable to fact-check. Not enough Wikipedia evidence.",
        }
    # 2. Retrieve most relevant chunks from FAISS
    relevant_chunks = document_store.search(claim, k=num_retrieved_chunks)
    context_for_llm = ""
    for chunk in relevant_chunks:
        context_for_llm += f"Source: {chunk['source']}\nContent: {chunk['text']}\n\n"
    if not context_for_llm:
        return {
            "response": "RAG Verdict: Unable to fact-check. No relevant Wikipedia context.",
        }
    # 3. Generate response with Gemini LLM
    prompt = f"""
You are an AI fact-checker. Given the following context from Wikipedia, output:
- A clear verdict: FAKE or REAL
- A confidence score (0-100, float, how sure you are about your verdict)
- A concise 2-3 line summary of the evidence for your verdict.
- Output format:
Verdict: FAKE/REAL
Confidence: <float>
Summary: <2-3 lines>
Only output these three fields, suitable for display in a browser extension.

Claim: "{claim}"

Context:
{context_for_llm}

Fact Check:
"""
    print("\nSending prompt to Gemini LLM...")
    try:
        response = llm_model.generate_content(prompt)
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            fact_check_result = response.candidates[0].content.parts[0].text
        else:
            fact_check_result = "Gemini LLM could not generate a response."
    except Exception as e:
        fact_check_result = f"Error generating response with Gemini LLM: {e}"
    return {
        "response": fact_check_result
    }

# --- Example Usage ---
if __name__ == "__main__":
    claims_to_check = [

        "Humans can breathe underwater indefinitely.",
        "Barack Obama was born in Kenya.", 
        "The Earth is flat.",

    ]

    for claim in claims_to_check:
        result = fact_check_wikipedia_only(claim)
        print("\n" + "="*50)
        print(f"Claim: {claim}")
        print(f"Fact Check Result:\n{textwrap.fill(result['response'], width=80)}")
        print("="*50 + "\n")