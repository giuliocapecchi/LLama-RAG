import os
import torch
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import time
from tqdm import tqdm
import re
import gradio as gr
from unidecode import unidecode

# ------------------------------------------------------------------------- #

# Load and process the PDF
loader = PyMuPDFLoader("IR Slides v1.0.pdf")
documents = loader.load()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Preprocess text to remove formulas and strange characters
def preprocess_text(text):
    # Rimuovi formule matematiche LaTeX
    text = re.sub(r'\$.*?\$', '', text)
    # Rimuovi caratteri non alfanumerici eccetto punteggiatura di base
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'"-]', '', text)
    # Normalizza i caratteri Unicode
    text = unidecode(text)
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def aggregate_short_documents(texts, min_length=50):
    aggregated_texts = []
    buffer = ""
    
    for doc in texts:
        if len(doc.page_content) < min_length:
            buffer += " " + doc.page_content
        else:
            if buffer:
                doc.page_content = buffer + " " + doc.page_content
                buffer = ""
            aggregated_texts.append(doc)
    
    if buffer:
        if aggregated_texts:
            aggregated_texts[-1].page_content += " " + buffer
        else:
            aggregated_texts.append(buffer)
    
    return aggregated_texts

# Applica la pre-elaborazione ai documenti
for doc in texts:
    doc.page_content = preprocess_text(doc.page_content)

# Aggrega i documenti corti
texts = aggregate_short_documents(texts)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="jxm/cde-small-v1")

# Create a vector store
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()

# ------------------------------------------------------------------------- #

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    print("CUDA is available. Working on GPU.")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Working on CPU.")

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
login(token=HF_TOKEN)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.repetition_penalty = 1.6  # Add repetition penalty

device = "cuda:0"

model.to(device)

# ------------------------------------------------------------------------- #

# Initialize the sentence-transformers model
embedder_model_name = "jxm/cde-small-v1"
embedder_model = SentenceTransformer(embedder_model_name)

def embedder(chunk):
    embeddings = embedder_model.encode(chunk, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# ------------------------------------------------------------------------- #

# Use the retriever to get the top_k_chunks for a given query
def search(query, retriever, k=10):
    results = retriever.invoke(query, k=k)
    return [result.page_content for result in results]

# ------------------------------------------------------------------------- #

def save_documents(texts):
    # delete all files in the vector_store directory
    for file in os.listdir("vector_store"):
        os.remove(os.path.join("vector_store", file))
    for i, doc in enumerate(texts):
        # print(f"Document {i+1}:\n{doc.page_content}\n{'-'*80}\n")
        with open(f"vector_store/document_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(doc.page_content)
# ------------------------------------------------------------------------- #

base_prompt = """You are an AI assistant for RAG. Your task is to understand the user question, and provide an answer using the provided contexts.

Your answers are correct, high-quality, and written by a domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

User question: {user_query}

Contexts:
{chunks_information}

Answer:
"""

# Example usage of the RAG system with the PDF
def answer_questions(questions):
    count = 0
    error = 0
    results = {}
    
    current_time = time.strftime("%m%d-%H%M%S")

    pbar = tqdm(questions, total=len(questions), desc="Answering questions...", unit="question")
    for q in pbar:
        top_k_chunks = search(q['question'], retriever, k=10)
        retrieved_chunks = [chunk for chunk in top_k_chunks]
        prompt = base_prompt.format(user_query=q['question'], chunks_information="\n".join(retrieved_chunks))
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config,
                num_beams=5,  # Use beam search for better results
                early_stopping=True,  # Stop early if all beams finish
            )

        # Exclude the prompt tokens from the generated output
        generated_tokens = outputs[0][len(encoding.input_ids[0]):]
        generated_unpreprocessed_sequence = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        match = re.search(r'\b[1-4]\b', generated_unpreprocessed_sequence)
        answer = match.group(0) if match else ""  # first number found or empty string

        with open(f"quiz/runs_basemodel/quiz_answers_{current_time}.txt", "a", encoding="utf-8") as f:
            f.write(f"Question: {q['question']}\nAnswer: {answer}\nCorrect answer:{q['correct']}\nGenerated unpreprocessed sequence: {generated_unpreprocessed_sequence}\n--------------------------------------------------------------------\n\n")

        results[q['question_id']] = answer

        if len(answer) != 1 or answer not in "1234":
            error += 1
        else:  # the format is correct, now check if the answer is correct
            if str(q['correct']) == answer:
                count += 1
        pbar.set_postfix(Corrects=f"{count}/{len(questions)}", Errors=error)
        
    print("-------------------------\tFINISHED RUN. Error count: ", error, "-------------------------")
    return results, count / len(questions) * 100

# Example questions
# questions = [
#     {"question": "", "correct": "1", "question_id": "1"},
#     # Add more questions as needed
# ]

# results, score = answer_questions(questions)
# print(f"Final score: {score}%")

# Call the function to save the documents
save_documents(texts)

def query_rag_model(user_query):
    top_k_chunks = search(user_query, retriever, k=10)
    retrieved_chunks = [chunk for chunk in top_k_chunks]
    prompt = base_prompt.format(user_query=user_query, chunks_information="\n".join(retrieved_chunks))
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
            num_beams=5,
            early_stopping=True,
        )
    generated_tokens = outputs[0][len(encoding.input_ids[0]):]
    generated_unpreprocessed_sequence = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return generated_unpreprocessed_sequence

iface = gr.Interface(
    fn=query_rag_model,
    inputs="text",
    outputs="text",
    title="RAG Model Query Interface",
    description="Ask questions to the RAG model and get answers based on the provided PDF context."
)

iface.launch()