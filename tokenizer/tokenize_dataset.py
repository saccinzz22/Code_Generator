import json
import random
import jsonlines#type:ignore
import sentencepiece as spm#type:ignore

random.seed(42)

SRC_MAX_LEN=768
TGT_MAX_LEN=512
TRAIN_RATIO=0.9

tokenizer=spm.SentencePieceProcessor()
tokenizer.load("tokenizer/coding_assistant.model")

train_writer=jsonlines.open("preprocessing/train.jsonl","w")
val_writer=jsonlines.open("preprocessing/val.jsonl", "w")

kept=0
dropped_src=0
dropped_tgt=0

with open("preprocessing/final_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        sample=json.loads(line)
        inp=sample["input"]
        trg=sample["target"]
        input_ids=tokenizer.encode(
            inp
        )
        target_ids=tokenizer.encode(
            trg
        )
        target_ids=[2]+target_ids+[3]
        if len(input_ids)>SRC_MAX_LEN:
            dropped_src+=1
            continue
        if len(target_ids)>TGT_MAX_LEN:
            dropped_tgt+=1
            continue
        out={
            "input_ids":input_ids,
            "target_ids":target_ids
        }
        if random.random()<TRAIN_RATIO:
            train_writer.write(
                out
            )
        else:
            val_writer.write(
                out
            )
        kept+=1
train_writer.close()
val_writer.close()

print("Kept",kept)
print("Dropped Source",dropped_src)
print("Dropped Target",dropped_tgt)
print()
print("Train")
print(
    "preprocessing/train.jsonl"
)
print()
print("Validation")
print(
    "preprocessing/val.jsonl"
)