import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
import joblib
from functools import lru_cache
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from ir_measures import calc_aggregate, P, R, SetF, nDCG

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('retrieval_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration Dataclass
@dataclass
class RetrieverConfig:
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    top_k: int = 10
    device: Optional[str] = None
    corpus_path: str = "hotpotqa/hotpotqa/corpus_10elems.jsonl"
    queries_path: str = "hotpotqa/hotpotqa/queries.jsonl"
    qrels_path: str = "hotpotqa/hotpotqa/qrels/dev.tsv"
    cache_dir: str = "cache"
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.cache_dir, exist_ok=True)

# Configuration and Environment Setup
def load_configuration(config_path: Optional[str] = None) -> RetrieverConfig:
    """
    Load configuration from file or return default configuration
    """
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return RetrieverConfig(**config_dict)
        return RetrieverConfig()
    except Exception as e:
        logger.error(f"Configuration loading error: {e}")
        raise

def validate_configuration(config: RetrieverConfig):
    """
    Validate configuration and environment
    """
    try:
        # Load environment variables
        load_dotenv()
        HF_TOKEN = os.getenv('HF_TOKEN')
        
        if not HF_TOKEN:
            raise ValueError("Hugging Face token is missing")
        
        login(token=HF_TOKEN)
        
        # Check required files
        required_files = [
            config.corpus_path,
            config.queries_path,
            config.qrels_path
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
        
        logger.info("Configuration validation successful")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

# Data Loading Functions
def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file with robust error handling
    """
    try:
        logger.info(f"Loading JSONL file from: {filepath}")
        with open(filepath, "r") as f:
            data = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(data)} records from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSONL file {filepath}: {e}")
        raise

# Embedding and Retrieval Functions
def create_embeddings(
    embedder_model: SentenceTransformer, 
    documents: List[Dict[str, Any]], 
    batch_size: int = 64,
    device: str = 'cpu'
) -> List[torch.Tensor]:
    """
    Create embeddings with memory-efficient batch processing
    """
    try:
        logger.info(f"Creating embeddings for {len(documents)} documents")
        embeddings = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Creating embeddings"):
            batch = documents[i:i+batch_size]
            # Use text from the document, fallback to empty string
            texts = [doc.get('text', '') for doc in batch]
            batch_embeddings = embedder_model.encode(
                texts, 
                convert_to_tensor=True, 
                device=device,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.cpu().numpy())
        
        return embeddings
    except Exception as e:
        logger.error(f"Embedding creation error: {e}")
        raise

def initialize_retriever(corpus: List[Dict[str, Any]], config: RetrieverConfig):
    """
    Initialize vector store and retriever with HotpotQA document compatibility
    """
    try:
        # Initialize sentence transformer model
        embedder_model = SentenceTransformer(config.embedder_model)
        embedder_model = embedder_model.to(config.device)
        
        # Create embeddings (ottimizzato per grandi corpus)
        corpus_embeddings = create_embeddings(
            embedder_model, 
            corpus, 
            batch_size=config.batch_size, 
            device=config.device
        )
        
        # Convert corpus to Langchain Documents
        documents = [
            Document(
                page_content=doc.get('text', ''),
                metadata={
                    'id': doc.get('_id', ''),
                    'title': doc.get('title', ''),
                    'url': doc.get('metadata', {}).get('url', '')
                }
            ) for doc in corpus
        ]
        
        # Create vector store USANDO gli embeddings precalcolati
        db = Chroma.from_documents(
            documents, 
            HuggingFaceEmbeddings(model_name=config.embedder_model),
            embedding=corpus_embeddings  # Passa gli embeddings precalcolati
        )
        retriever = db.as_retriever(search_kwargs={"k": config.top_k})
        
        return retriever
    except Exception as e:
        logger.error(f"Retriever initialization error: {e}")
        logger.error(f"Corpus sample: {corpus[:2]}")
        raise

def search(query: str, retriever, k: int = 10) -> List[str]:
    """
    Perform search and return document IDs
    """
    try:
        logger.debug(f"Searching for query: {query[:30]}...")
        results = retriever.invoke(query)
        doc_ids = [result.metadata.get('id', '') for result in results]
        logger.debug(f"Retrieved {len(doc_ids)} results")
        return doc_ids
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise

# Qrels Loading Function
def load_qrels(qrels_path: str) -> pd.DataFrame:
    """
    Load qrels with appropriate columns for HotpotQA
    """
    try:
        # Assuming you have a qrels file with columns: query_id, doc_id, relevance
        qrels = pd.read_csv(qrels_path, sep='\t', header=None, names=["query_id", "doc_id", "relevance"])
        logger.info(f"Loaded qrels with shape: {qrels.shape}")
        return qrels
    except Exception as e:
        logger.error(f"Qrels loading error: {e}")
        raise

# Retrieval Evaluator Class
class RetrievalEvaluator:
    def __init__(self, retriever, cache_dir: str = 'cache'):
        self.retriever = retriever
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    @lru_cache(maxsize=1000)
    def cached_search(self, query: str, k: int = 10):
        return search(query, self.retriever, k)
    
    def save_metrics(self, metrics: Dict[Any, float]):
        """Save evaluation metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.cache_dir, f'metrics_{timestamp}.pkl')
            joblib.dump(metrics, filename)
            logger.info(f"Metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Metrics saving error: {e}")

# Evaluation Functions
def evaluate_retrieval_system(
    queries: List[Dict[str, Any]], 
    qrels: pd.DataFrame, 
    retriever, 
    k: int = 10
) -> Dict[Any, float]:
    """
    Comprehensive retrieval system evaluation
    """
    try:
        results = []
        logger.info(f"Evaluating retrieval system with {len(queries)} queries")

        for query in tqdm(queries, desc="Evaluating queries"):
            query_id = query.get('_id', '')
            query_text = query.get('text', '')
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
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Retrieval system evaluation error: {e}")
        raise

def print_evaluation_results(metrics: Dict[Any, float]):
    """
    Print evaluation metrics in a formatted way
    """
    print("\n--- Retrieval System Evaluation Results ---")
    print(f"Precision@10: {metrics[P@10]:.4f}")
    print(f"Recall@10: {metrics[R@10]:.4f}")
    print(f"F1@10: {metrics[SetF@10]:.4f}")
    print(f"nDCG@10: {metrics[nDCG@10]:.4f}")
    print("-------------------------------------------\n")

def main(config_path: Optional[str] = None):
    """
    Main execution function for retrieval system
    """
    try:
        # Load and validate configuration
        config = load_configuration(config_path)
        validate_configuration(config)
        
        logger.info(f"Starting retrieval system with config: {asdict(config)}")
        
        # Load data
        corpus = load_jsonl(config.corpus_path)
        queries = load_jsonl(config.queries_path)
        qrels = load_qrels(config.qrels_path)
        
        # Initialize retriever
        retriever = initialize_retriever(corpus, config)
        
        # Evaluate
        evaluator = RetrievalEvaluator(retriever, config.cache_dir)
        metrics = evaluate_retrieval_system(
            queries, 
            qrels, 
            retriever, 
            k=config.top_k
        )
        
        # Log and save metrics
        evaluator.save_metrics(metrics)
        
        # Print results
        print_evaluation_results(metrics)
        
    except Exception as e:
        logger.error(f"Retrieval system execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()