import transformers
import json
from datasets import Features, Value, load_dataset
import pandas as pd

model = transformers.AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

# Carica il dataset da un file JSON locale
corpus = load_dataset("json", data_files="hotpotqa/hotpotqa/corpus.jsonl")
# Droppa 'metadata' dal dataset, contiene solo url e del documento
corpus = corpus.remove_columns(['metadata']) 

print(corpus)
print(corpus['train'][:5])  # Mostra le prime 5 righe del corpus

# Read the qrels file
qrels_file = "hotpotqa/hotpotqa/qrels/dev.tsv"
qrels = pd.read_csv(qrels_file, sep='\t', header=None, names=["query_id", "corpus_id", "score"])

# Print the first 5 entries of the qrels
print(qrels.head())

# Paths to the input and output files
input_file = "hotpotqa/hotpotqa/queries.jsonl"
output_file = "hotpotqa/hotpotqa/queries_fixed.jsonl"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        # Remove `supporting_facts` from metadata
        if "supporting_facts" in data["metadata"]:
            del data["metadata"]["supporting_facts"]
        # Write the modified data back to a new file
        outfile.write(json.dumps(data) + "\n")

# Define the schema without `supporting_facts`
features = Features({
    "_id": Value("string"),
    "text": Value("string"),
    "metadata": {
        "answer": Value("string"),
    }
})

# Load the preprocessed dataset
queries = load_dataset("json", data_files="hotpotqa/hotpotqa/queries_fixed.jsonl", features=features)

print(queries)
print(queries['train'][:5])  # Display the first 5 rows

query_prefix = "search_query: "
document_prefix = "search_document: "

def process_ex_document(ex: dict) -> dict:
  ex["text"] = f"{ex['title']} {ex['text']}"
  return ex

# Select a few queries and their relevant documents based on qrels
selected_queries = queries['train'].select(range(5))
selected_query_ids = selected_queries['_id']
relevant_docs_ids = qrels[qrels['query_id'].isin(selected_query_ids)]['corpus_id'].unique()

# Debugging: Print relevant document IDs
print("Relevant document IDs:", relevant_docs_ids)

# Filter the corpus to include only relevant documents
filtered_corpus = corpus['train'].filter(lambda doc: doc['_id'] in relevant_docs_ids)

# Debugging: Print the filtered corpus
print("Filtered corpus:", filtered_corpus)

filtered_corpus = filtered_corpus.map(process_ex_document)["text"]

# Ensure filtered_corpus is not empty before tokenizing
if len(filtered_corpus) == 0:
    raise ValueError("Filtered corpus is empty. Check the document IDs and qrels.")

filtered_corpus = tokenizer(
    [document_prefix + doc for doc in filtered_corpus],
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
)

# # 1. gather embeddings
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
model.to(device)
filtered_corpus = filtered_corpus.to(device)

from tqdm.autonotebook import tqdm

batch_size = 32

dataset_embeddings = []
for i in tqdm(range(0, len(filtered_corpus["input_ids"]), batch_size)):
    filtered_corpus_batch = {k: v[i:i+batch_size] for k,v in filtered_corpus.items()}
    with torch.no_grad():
        dataset_embeddings.append(
            model.first_stage_model(**filtered_corpus_batch)
        )

dataset_embeddings = torch.cat(dataset_embeddings)

# # 2. Embed in context
docs = filtered_corpus

with torch.no_grad():
  doc_embeddings = model.second_stage_model(
      input_ids=docs["input_ids"],
      attention_mask=docs["attention_mask"],
      dataset_embeddings=dataset_embeddings,
  )
doc_embeddings /= doc_embeddings.norm(p=2, dim=1, keepdim=True)

queries = selected_queries["text"]
queries = tokenizer(
    [query_prefix + query for query in queries],
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
).to(device)

with torch.no_grad():
  query_embeddings = model.second_stage_model(
      input_ids=queries["input_ids"],
      attention_mask=queries["attention_mask"],
      dataset_embeddings=dataset_embeddings,
  )
query_embeddings /= query_embeddings.norm(p=2, dim=1, keepdim=True)

import seaborn as sns

sns.heatmap((doc_embeddings @ query_embeddings.T).cpu(), cmap="jet")

# Define synthetic documents and a query
synthetic_docs = [
    {"title": "Document 1", "text": "This is the first synthetic document about machine learning."},
    {"title": "Document 2", "text": "This document discusses the applications of artificial intelligence."},
    {"title": "Document 3", "text": "Here we talk about the advancements in natural language processing."},
    {"title": "Document 4", "text": "This document is about the history of computer science."},
    {"title": "Document 5", "text": "This document covers the basics of deep learning and neural networks."}
]

synthetic_query = "What are the applications of artificial intelligence?"

# Process synthetic documents
synthetic_docs = [process_ex_document(doc) for doc in synthetic_docs]
synthetic_docs_texts = [doc["text"] for doc in synthetic_docs]
synthetic_docs_tokenized = tokenizer(
    [document_prefix + doc for doc in synthetic_docs_texts],
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
).to(device)

# Generate embeddings for synthetic documents
with torch.no_grad():
    synthetic_doc_embeddings = model.second_stage_model(
        input_ids=synthetic_docs_tokenized["input_ids"],
        attention_mask=synthetic_docs_tokenized["attention_mask"],
        dataset_embeddings=dataset_embeddings,
    )
synthetic_doc_embeddings /= synthetic_doc_embeddings.norm(p=2, dim=1, keepdim=True)

# Tokenize and embed the synthetic query
synthetic_query_tokenized = tokenizer(
    query_prefix + synthetic_query,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    synthetic_query_embedding = model.second_stage_model(
        input_ids=synthetic_query_tokenized["input_ids"],
        attention_mask=synthetic_query_tokenized["attention_mask"],
        dataset_embeddings=dataset_embeddings,
    )
synthetic_query_embedding /= synthetic_query_embedding.norm(p=2, dim=1, keepdim=True)

# Calculate similarity
similarity_scores = (synthetic_doc_embeddings @ synthetic_query_embedding.T).cpu().numpy().flatten()

# Print similarity list
print("Query:", synthetic_query)
print("Similarity scores with synthetic documents:")
for i, score in enumerate(similarity_scores):
    print(f"Document {i+1}: {score:.4f} - {synthetic_docs[i]['text']}")


print("Query IDs in queries:", queries['train']["_id"][:5])
print("Query IDs in qrels:", qrels["query_id"].unique()[:5])


def print_query_with_documents(query_id, queries, corpus, qrels):
    # Trova la query corrispondente
    query = queries.filter(lambda q: q["_id"] == query_id)["text"][0]
    print(f"Query: {query}\n")

    # Trova i documenti rilevanti per la query nel qrels
    relevant_doc_ids = qrels[qrels["query_id"] == query_id]["corpus_id"].tolist()

    # Filtra il corpus per ottenere i documenti rilevanti
    relevant_docs = [doc for doc in corpus if doc["_id"] in relevant_doc_ids][:5]  # Prendi al massimo 5 documenti

    # Stampa i documenti rilevanti
    print("Relevant Documents:")
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"Document {i}: {doc['title']}\n{doc['text']}\n")


# Seleziona una query casuale
query_id_to_test = queries['train']["_id"][0]  # Scegli la prima query per esempio

# Converto corpus['train'] in lista per usarlo facilmente
corpus_list = corpus["train"]

# Stampa la query e i suoi documenti rilevanti
print_query_with_documents(query_id_to_test, queries['train'], corpus_list, qrels)
