import os
import json
from huggingface_hub import login
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ir_measures import calc_aggregate, P, R, SetF, nDCG
import pandas as pd
from tqdm import tqdm
import torch

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
print(f"HF_TOKEN loaded: {HF_TOKEN is not None}")
login(token=HF_TOKEN)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    print("CUDA is available. Working on GPU.")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = "cuda:0"
else:
    print("CUDA not available. Working on CPU.")
    device = "cpu"

    
# Initialize the sentence-transformers model
embedder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading embedder model: {embedder_model_name}")
embedder_model = SentenceTransformer(embedder_model_name)
embedder_model = embedder_model.to(device)  # Move model to GPU

# Load the corpus
def load_jsonl(filepath):
    print(f"Loading JSONL file from: {filepath}")
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} records from {filepath}")
    return data

corpus = load_jsonl("hotpotqa/hotpotqa/corpus.jsonl")
queries = load_jsonl("hotpotqa/hotpotqa/queries.jsonl")

# Load the qrels (relevance judgments)
qrels = pd.read_csv("hotpotqa/hotpotqa/qrels/dev.tsv", sep='\t', header=None, names=["query_id", "doc_id", "relevance"])
print(f"Loaded qrels with shape: {qrels.shape}")

# Create embeddings for the corpus using batch processing
def create_embeddings(embedder_model, documents, batch_size=64):
    print(f"Creating embeddings for {len(documents)} documents with batch size {batch_size}")
    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Creating embeddings"):
        batch = documents[i:i+batch_size]
        texts = [doc['text'] for doc in batch]
        batch_embeddings = embedder_model.encode(texts, convert_to_tensor=True, device=device)
        embeddings.extend(batch_embeddings)
    return embeddings

corpus_embeddings = create_embeddings(embedder_model, corpus)
print(f"Created embeddings for corpus")

# Create a vector store
embeddings = HuggingFaceEmbeddings(model_name=embedder_model_name)
db = Chroma.from_documents(corpus, embeddings)
retriever = db.as_retriever()
print("Vector store and retriever initialized")

# Save the Chroma store
db.save("chroma_store")
print("Chroma store saved")

# Function to embed a chunk of text
def embedder(chunk):
    print(f"Embedding chunk of text: {chunk[:30]}...")
    return embedder_model.encode(chunk, convert_to_tensor=True, device=device).cpu().numpy()

# Use the retriever to get the top_k_chunks for a given query
def search(query, retriever, k=10):
    print(f"Searching for query: {query[:30]}...")
    results = retriever.invoke(query, k=k)
    print(f"Retrieved {len(results)} results")
    return [result.metadata['id'] for result in results]  # Ensure doc IDs are returned for proper evaluation

# Evaluate the retrieval system
def evaluate_retrieval_system(queries, qrels, retriever, k=10):
    results = []
    print(f"Evaluating retrieval system with {len(queries)} queries")

    for query in tqdm(queries, desc="Evaluating queries"):
        query_id = query['id']
        query_text = query['text']
        relevant_docs = qrels[qrels['query_id'] == query_id]['doc_id'].tolist()

        retrieved_docs = search(query_text, retriever, k=k)

        for rank, doc_id in enumerate(retrieved_docs, start=1):
            results.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'rank': rank,
                'relevance': 1 if doc_id in relevant_docs else 0
            })

    metrics = calc_aggregate(results, [P@10, R@10, SetF@10, nDCG@10])
    print(f"Evaluation metrics: {metrics}")
    return metrics

# Main function to run the evaluation
def main():
    # Check if the Chroma store exists
    if os.path.exists("chroma_store"):
        print("Loading existing Chroma store")
        db = Chroma.load("chroma_store")
        retriever = db.as_retriever()
    else:
        print("Creating new Chroma store")
        # Create embeddings for the corpus
        corpus_embeddings = create_embeddings(embedder_model, corpus)
        print(f"Created embeddings for corpus")        
        # Create a vector store using the embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedder_model_name)
        db = Chroma.from_documents(corpus, embeddings, embeddings=corpus_embeddings)
        retriever = db.as_retriever()
        print("Vector store and retriever initialized")
        db.save("chroma_store")
        print("Chroma store saved")
    print("Starting evaluation")
    metrics = evaluate_retrieval_system(queries, qrels, retriever, k=10)
    print(f"Precision@10: {metrics[P@10]:.4f}")
    print(f"Recall@10: {metrics[R@10]:.4f}")
    print(f"F1@10: {metrics[SetF@10]:.4f}")
    print(f"nDCG@10: {metrics[nDCG@10]:.4f}")

if __name__ == "__main__":
    main()
