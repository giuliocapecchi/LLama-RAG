import torch
import os 
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ------------------------------------------------------------------------- #

# load the model from the disk in 'trained-model'

#MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda:0"
model = AutoModelForCausalLM.from_pretrained("trained-model").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

generation_config = model.generation_config
generation_config.max_new_tokens = 50
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


def generate_response(question: str) -> str:
    prompt = f"""
        <human>: {question}
        <assistant>:
        """.strip()
    encoding = tokenizer(prompt, return_tensors="pt") .to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start)
    return response[response_start + len(assistant_start) : ].strip()


# ------------------------------------------------------------------------- #

generation_config = model.generation_config
generation_config.max_new_tokens = 100
generation_config.temperature = 0.1
generation_config.top_p = 1.0
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

import json

def extract_questions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    questions = []
    for item in data:
        q = item.get("question", "")
        a = item.get("answers", "")
        # 'a' è una lista di stringhe, aggiungi ad ognuno 1, 2, 3, 4
        for i in range(len(a)):
            a[i] = f"{(1 + i)}) {a[i]}"
        a = " ".join(a)
        qplusa = f"{q} {a}"
        question = {
            "question": qplusa,
            "correct": item.get("correct", ""),
            "category": item.get("category", ""),
            "question_id": item.get("question_id", "")
        }
        questions.append(question)
    return questions


questions = extract_questions('quiz/quiz.json')
# count all the meme questions
meme_questions = [q for q in questions if int(q['category']) == 1]

import time
from tqdm import tqdm
import re

def generate_response(question: str) -> str:
    prompt = f"""
        <human>: {question}
        <assistant>:
        """.strip()
    encoding = tokenizer(prompt, return_tensors="pt") .to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start)
    return response[response_start + len(assistant_start) : ].strip()


def answer_questions():
    count = 0
    meme = 0
    error = 0
    results = {}
    
    current_time = time.strftime("%m%d-%H%M%S")

    pbar = tqdm(questions, total=len(questions), desc="Answering questions...", unit="question")
    for q in pbar:
        base_prompt = "You are an AI assistant. Your task is to provide the correct answer to the following question by outputting only the number (1, 2, 3, or 4) corresponding to the correct option. Do not include any additional text in your response."
        prompt = f"{base_prompt}\n{q['question']}".strip()
        answer = ""
        generated_unpreprocessed_sequence = ""
        for _ in range(3):  # se il formato della risposta non è corretto, riprova fino a 3 volte
            generated_unpreprocessed_sequence = generate_response(prompt).strip().lower()
            # extract the first number to appear in the answer
            match = re.search(r'\b[1-4]\b', generated_unpreprocessed_sequence)
            answer = match.group(0) if match else "" # first number found or empty string
            # print("\nAnswer: ",answer)
            if len(answer) == 1 and answer in "1234":
                break  # Valid answer format

        # write question and answer to a file
        with open(f"quiz/runs_finetunedmodel/quiz_answers_{current_time}.txt", "a", encoding="utf-8") as f:
            f.write(f"Question: {q['question']}\nAnswer: {answer}\nCorrect answer:{q['correct']}\nGenerated unpreprocessed sequence: {generated_unpreprocessed_sequence}\n--------------------------------------------------------------------\n\n")

        results[q['question_id']] = answer

        if len(answer) != 1 or answer not in "1234":
            error += 1
        else: # the format is correct, now check if the answer is correct
            if str(q['correct']) == answer:
                if int(q['category']) == 0: # normal question
                    count += 1
                elif int(q['category']) == 1: # meme question
                    meme += 1
                else:
                    print("Error : category not valid for prompt - >",prompt)

        # aggiungi correct e meme alla tqdm bar
        pbar.set_postfix(Corrects=f"{count}/{len(questions)-len(meme_questions)}", Meme=f"{meme}/{len(meme_questions)}", Errors=error)
        
    
    print("-------------------------\tFINISHED RUN. Error count: ",error, "-------------------------")
    return results, count/60*100, meme/10*100


count = 0
meme = 0
iterations = 10
results = {}

print(f"Total normal questions: {len(questions)-len(meme_questions)}\tTotal meme questions: {len(meme_questions)}")

for i in range(iterations):
    results_new, count_i, meme_i = answer_questions()
    count += count_i
    meme += meme_i
    print(f"Epoch : {i}) Medium score: {count_i}\tMedium meme score: {meme_i}\n\n")
    results[i] = results_new

# print medium score
print("----------------------------------------------------------------------------------")
print(f"Medium score: {count/iterations*100}%\tMedium meme score: {meme/iterations*100}%")

# pickle results
import pickle

current_time = time.strftime("%m%d-%H%M%S")
with open(f"quiz/runs_finetunedmodel/quiz_results_{current_time}.pkl", "wb") as f:
    pickle.dump(results, f)