import json
import numpy as np
import sentencepiece as spm#type:ignore

tokenizer=spm.SentencePieceProcessor()

tokenizer.load(
    "tokenizer/coding_assistant.model"
)

total_examples=0

total_input_tokens=0

total_target_tokens=0

input_lengths=[]

target_lengths=[]

with open(
    "preprocessing/final_dataset.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        sample=json.loads(line)

        inp=sample["input"]

        trg=sample["target"]

        input_ids=tokenizer.encode(inp)

        target_ids=tokenizer.encode(trg)

        input_len=len(input_ids)

        target_len=len(target_ids)

        total_examples+=1

        total_input_tokens+=input_len

        total_target_tokens+=target_len

        input_lengths.append(input_len)

        target_lengths.append(target_len)

input_lengths=np.array(input_lengths)

target_lengths=np.array(target_lengths)

print("="*50)

print("TOTAL EXAMPLES")

print("="*50)

print(total_examples)

print()

print("="*50)

print("INPUT TOKENS")

print("="*50)

print("Total :",total_input_tokens)

print("Average :",round(input_lengths.mean(),2))

print("Median :",int(np.median(input_lengths)))

print("Max :",input_lengths.max())

print()

print("="*50)

print("TARGET TOKENS")

print("="*50)

print("Total :",total_target_tokens)

print("Average :",round(target_lengths.mean(),2))

print("Median :",int(np.median(target_lengths)))

print("Max :",target_lengths.max())

print()

print("="*50)

print("TOTAL TOKENS")

print("="*50)

print(total_input_tokens+total_target_tokens)