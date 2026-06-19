import json
import numpy as np
import sentencepiece as spm#type:ignore

tokenizer=spm.SentencePieceProcessor()
tokenizer.load(
    "tokenizer/coding_assistant.model")

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
        input_lengths.append(
            len(input_ids)
        )
        target_lengths.append(
            len(target_ids)
        )
input_lengths=np.array(input_lengths)
target_lengths=np.array(target_lengths)

print("INPUT")
for p in [50,75,90,95,99]:
    print(
        f"{p}% :",
        int(
            np.percentile(
                input_lengths,
                p
            )
        )
    )
print("MAX :", input_lengths.max())
print()

print("TARGET")
for p in [50,75,90,95,99]:
    print(
        f"{p}% :",
        int(
            np.percentile(
                target_lengths,
                p
            )
        )
    )
print("MAX :",target_lengths.max())