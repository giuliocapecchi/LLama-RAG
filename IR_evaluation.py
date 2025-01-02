from ir_measures import *
from collections import defaultdict
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch
import os
import json
import random



def evaluate_retrieval(
    embeddings_model,
    corpus: List[str],
    queries: List[Dict],
    qrels: Dict[str, Dict[str, int]],
    dataset_embeddings,
    k_values: List[int] = [1, 3, 5, 10],
    device: str = "cuda"
) -> Dict:
    """
    Evaluate retrieval performance using standard IR metrics.
    
    Args:
        embeddings_model: The SentenceTransformer model
        corpus: List of documents
        queries: List of dictionaries containing 'question_id' and 'question'
        qrels: Dictionary mapping query_ids to document relevance judgments
        dataset_embeddings: Contextualized embeddings of the minicorpus
        k_values: List of k values for which to compute metrics
        device: Device to use for computations
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Embed documents and queries
    doc_embeddings = embeddings_model.encode(
        corpus,
        prompt_name="document",
        dataset_embeddings=dataset_embeddings,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    
    query_embeddings = embeddings_model.encode(
        [q['question'] for q in queries],
        prompt_name="query",
        dataset_embeddings=dataset_embeddings,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    
    # Initialize run dictionary for storing rankings
    run = defaultdict(dict)
    
    # For each query
    for query_idx, query in enumerate(queries):
        query_id = query['question_id']
        
        # Get top-k results for maximum k
        max_k = max(k_values)
        print("max_k: ", max_k)
        scores_and_indices = embeddings_model.similarity(
            query_embeddings[query_idx:query_idx+1],
            doc_embeddings,
       )
        
        scores_and_indices = scores_and_indices.topk(max_k)

        retrieved_doc_indices = scores_and_indices[1].cpu().numpy()[0]
        
        # Store rankings
        for rank, doc_idx in enumerate(retrieved_doc_indices):
            run[query_id][str(doc_idx)] = max_k - rank  # Higher score for higher ranks
    

    metrics = [AP(rel=2), nDCG, nDCG@10, Recall(rel=2)@1000]

    # Calculate metrics using calc_aggregate
    results = {}
    for metric in metrics:
        results[str(metric)] = calc_aggregate([metric], qrels, run)
    
    return results


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from a txt tab separated file.
    Expected format: 
    query_id document_id relevance iteration
    """
    qrels = defaultdict(dict)
    with open(qrels_path, 'r') as file:
        for line in file:
            query_id, doc_id, relevance, _ = line.strip().split()
            qrels[query_id][doc_id] = int(relevance)
    return dict(qrels)
    


def extract_questions(file_path):
    """
    Extracts questions from a JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        return [
            {
                "question_id": item.get("question_id", ""),
                "question": item.get("question", "")
            }
            for item in data
        ]
    


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings_model = SentenceTransformer(
    "jxm/cde-small-v1",
    trust_remote_code=True,
    ).to(device)

    corpus = []
    # for each document in the 'chunks' folder, we append its text to 'corpus'
    for file in sorted(os.listdir("chunks")):
        with open(os.path.join("chunks", file), "r", encoding="utf-8") as f:
            corpus.append(f.read())

    print("len del corpus: ", len(corpus))

    qrels = load_qrels("evaluation/qrels.tsv")
    print(f"Loaded {len(qrels)} QRELS.")

    print("QRELS : ", qrels)

    queries = extract_questions("evaluation/open_questions.json")
    print(f"Loaded {len(queries)} questions.")


    minicorpus_size = embeddings_model[0].config.transductive_corpus_size # 512
    random.seed(424242)
    minicorpus_docs = random.choices(corpus, k=minicorpus_size) # oversampling is okay
    assert len(minicorpus_docs) == minicorpus_size # We must use exactly this many documents in the minicorpus

    dataset_embeddings = embeddings_model.encode(
        [doc for doc in minicorpus_docs],
        prompt_name="document",
        convert_to_tensor=True,
        show_progress_bar=True
    )

    metrics = evaluate_retrieval(
        embeddings_model=embeddings_model,
        corpus=corpus,
        queries=queries,
        qrels=qrels, 
        dataset_embeddings=dataset_embeddings
    )
    
    # Print results
    print("\nRetrieval Evaluation Results:")
    for metric_name, values in metrics.items():
        if isinstance(values, dict):
            for k, score in values.items():
                print(f"  @{k}: {score:.3f}")
        else:
            print(f"\n{metric_name}: {values:.3f}")