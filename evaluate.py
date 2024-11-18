from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from tqdm import tqdm
import csv
max_seq_length = 2048 # Choose any
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "autodl-tmp/outputs/checkpoint-9500", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map = "cuda",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

prompt = """You are a great mathematician and you are tasked with finding if an answer to a given maths question is correct or not. Yout response should be 'True' if correct, otherwise 'False'. Below is Question and Answer.



### Question:
{}

### Answer:
{}

### Explainaition:
{}

### Output:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp")

test_dataset = dataset['test']

csv_file = 'results2.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['ID', 'is_correct'])

    for i in tqdm(range(len(test_dataset))):
    #for i in tqdm(range(2)):
        sample_ques = test_dataset['question'][i]
        sample_ans = test_dataset['answer'][i]
        sample_expl = test_dataset['solution'][i]

        input_prompt = prompt.format(
                sample_ques, # ques
                sample_ans, # given answer,
                sample_expl, # explaination
                "", # output - leave this blank for generation! LLM willl generate is it is True or False
            )

        inputs = tokenizer(
        [
            input_prompt
        ], return_tensors = "pt").to("cuda")

        input_shape = inputs['input_ids'].shape
        input_token_len = input_shape[1] # 1 because of batch
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        # you can get the whole generated text by uncommenting the below line
        # text_generated = tokenizer.batch_decode([outputs, skip_special_tokens=True)

        response = tokenizer.batch_decode([outputs[0][input_token_len:]], skip_special_tokens=True)[0]
        writer.writerow([i, response])