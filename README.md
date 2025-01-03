# Retrieval-Augmented Generation (RAG) for chatting with your PDFs using LLama 3.2 3B

This project implements a **Retrieval-Augmented Generation (RAG)** system, a modern technique that integrates document retrieval and answer generation. The RAG approach enhances the performance of large language models (LLMs) by incorporating external information to improve the model's ability to answer queries outside its original training data. This is especially beneficial for businesses that cannot afford to build a ChatGPT-like LLM from scratch. RAG provides a faster and more efficient alternative than re-training the model, as it tries to expand the model knowledge by simply providing relevant documents as data sources to the user's queries.

---

## Pipeline Overview

<div align="center">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF logo">
</div>

The proposed pipeline will:
1. Split the documents into smaller chunks (with special handling for *mathematical formulas*).
2. Embed these chunks using a model from Hugging Face (`cde-small-v1`), which is currently (as of January 2024) the smallest top-ranked in the [Massive Text Embedding Benchmark (MTEB)](!https://huggingface.co/spaces/mteb/leaderboard) Leaderboard.
3. Initialize a (quantized) local copy of `LLaMA 3.2 3B` for answer generation.
4. Provide access to the model through a `Gradio` interface.

---

## Workflow Overview
### 1. **PDF Preprocessing**
   - Preprocess PDFs using `Unstructured` and `Nougat` for extracting structured elements and handling complex pages (e.g., with formulas).
   - Partition PDF content into chunks for embedding generation.

### 2. **Document Embeddings**
   - Generate embeddings for the document chunks using the `cde-small-v1` model.

### 3. **LLM Integration**
   - Load the `LLaMA 3.2 3B` model, optimized with 4-bit quantization for local execution.
   - Integrate embeddings into the RAG pipeline to perform retrieval and answer generation.

### 4. **Query Execution**
   - Perform queries against the RAG pipeline to retrieve answers based on the embedded document context.

### 5. **Evaluation**
   - Assess retrieval and generation quality using IR metrics like nDCG, Recall, and Reciprocal Rank (RR).

---

## Gradio Interface

<div align="center"><img src="https://www.gradio.app/_app/immutable/assets/gradio.CHB5adID.svg" alt="Gradio Logo" width="200"></div>

To make the RAG system accessible, a Gradio interface is included. You can query the model interactively with your custom prompts and receive context-based answers.

---

## Evaluation Metrics
The RAG system is evaluated using the following metrics:
- **AP (Average Precision)**: Evaluates ranking accuracy.
- **nDCG (Normalized Discounted Cumulative Gain) and nDCG@5**: Measures ranking quality based on relevance.
- **Precision**: The ratio of relevant documents retrieved to the total number of documents retrieved.
- **Recall**: Captures the proportion of relevant documents retrieved.
- **RR (Reciprocal Rank)**: Indicates the rank position of the first relevant document.

---

## Steps to Run the Project

Just follow the `RAG.ipynb` notebook. All the steps are extensively explained there.

---

## Conclusion
This project showcases the implementation of a RAG pipeline for enhanced query answering. By combining document retrieval with language model generation, it demonstrates an efficient approach. Please explore the notebook for more code and insights!