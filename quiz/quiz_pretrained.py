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

MODEL_NAME = "meta-llama/Llama-3.2-3B" # 3B

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

def print_trainable_parameters (model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters ():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.1
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

device = "cuda:0"

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

from tqdm import tqdm

@profile
def answer_questions():
    count = 0
    meme = 0
    error = 0
    # count all the meme questions
    meme_questions = [q for q in questions if int(q['category']) == 1]
    print(f"Total normal questions: {len(questions)-len(meme_questions)}\tTotal meme questions: {len(meme_questions)}")
    current_time = time.strftime("%m%d-%H%M%S")

    for q in tqdm(questions, total=len(questions), desc="Answering questions...", unit="question"):
        base_prompt = "Respond ONLY with the number corresponding to the correct option (1, 2, 3, or 4). Use only one token for your answer."
        prompt = f"{base_prompt} {q['question']}"
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        
        answer = ""
        for _ in range(3):  # Try up to 3 times to get a valid answer
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                    generation_config=generation_config,
                )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            # extract "answer : 1, 2, 3, 4" from the generated text
            answer = answer.lower().split("answer :")[-1].strip()
            answer = answer.split("\n")[-1]  # Prendi l'ultima riga nel caso ci siano più righe
            print("Answer: ",answer)
            if len(answer) == 1 and answer in "1234":
                break  # Valid answer format

        # write question and answer to a file
        with open(f"quiz/runs/quiz_answers_{current_time}.txt", "a", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\nAnswer: {answer}\n\n")

        if len(answer) != 1 or answer not in "1234":
            error += 1
            tqdm.write(f"Correct: {count}/{len(questions)-len(meme_questions)} | Meme questions: {meme}/{len(meme_questions)} | Formatting errors: {error}")
            continue

        if str(q['correct']) == answer:
            if int(q['category']) == 0:
                count += 1
            elif int(q['category']) == 1:
                meme += 1
            else:
                print("Error : category not valid for prompt - >",prompt)

        # aggiungi correct e meme alla tqdm bar
        tqdm.write(f"Corrects: {count}/{len(questions)-len(meme_questions)} | Meme questions: {meme}/{len(meme_questions)} | Formatting errors: {error}")
        
    
    print("-------------------------\nFINISHED RUN. Error count: ",error)
    return count/40, meme/5


count = 0
meme = 0
iterations = 1

for i in range(iterations):
    count_i, meme_i = answer_questions()
    count += count_i
    meme += meme_i
    print(f"{i}) Medium score: {count/iterations}")
    print(f"{i}) Medium meme score: {meme/iterations}")

# print medium score
print("-------------------------------------------")
print(f"Medium score: {count/iterations}")
print(f"Medium meme score: {meme/iterations}")