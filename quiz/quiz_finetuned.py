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

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print("CUDA is available. Working on GPU.")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Working on CPU.")

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
login(token=HF_TOKEN)

# ------------------------------------------------------------------------- #

# load the model from the disk in 'trained-model'

#MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
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
            "category": item.get("category", "")
        }
        questions.append(question)
    return questions


questions = extract_questions('quiz/merged_quiz.json')


import time

def profile(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        ms = (end - start) * 1000
        print(f"{f.__name__} ({ms:.3f} ms)")
        return result
    return f_timer

@profile
def answer_questions():
    count = 0
    meme = 0
    error = 0
    for q in questions:
        base_prompt = "Answer only with the letter of the correct option, with ONE TOKEN ONLY."
        prompt = f"{base_prompt} Question: {q['question']}"
        #print("\n\nPROMPT: ",prompt)
        answer = generate_response(prompt)
        # extract the correct answer from the output, will be in 'Answer: b' form
        answer = answer.lower()
        try : 
            answer = answer.split("answer: ")[1].strip()
        except:
            print("ERROR: ",q['question'],"\nANSWER", answer)
            error += 1
            continue
        print("CORRECT: ",q['correct'], "\tMODEL ANSWER: ",answer)
        if str(q['correct']) == answer:
            if int(q['category']) == 0:
                print("Risposta corretta")
                count += 1
            elif int(q['category']) == 1:
                print("Risposta corretta ma meme")
                meme += 1
            else:
                print("Error : category not valid for prompt - >",prompt)
    
    print("-------------------------\nFINISHED RUN. Error count: ",error)
    return count/40, meme/5


count = 0
meme = 0

for i in range(1):
    count_i, meme_i = answer_questions()
    count += count_i
    meme += meme_i
    print(f"{i}) Medium score: {count/10}")
    print(f"{i}) Medium meme score: {meme/10}")

# print medium score
print("-------------------------------------------")
print(f"Medium score: {count/10}")
print(f"Medium meme score: {meme/10}")