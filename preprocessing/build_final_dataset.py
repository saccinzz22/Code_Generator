import json
import jsonlines#type:ignore
import pandas as pd
from pathlib import Path
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning
)

sys.set_int_max_str_digits(0)

root=Path("datasets")
mbpp=root/"Code_generation"/"mbpp.jsonl"
apps=root/"Code_generation"/"APPS+.json"
ds1000=root/"DS1000"/"ds1000.jsonl"
humaneval=root/"humaneval"
humanevalplus=root/"humaneval+"
output="preprocessing/final_dataset.jsonl"

writer=jsonlines.open(output,"w")

seen=set()
def write_sample(inp,trg):
    inp=str(inp).strip()
    trg=str(trg).strip()
    if len(inp)==0 or len(trg)==0:
        return
    key=(inp,trg)
    if key in seen:
        return
    seen.add(key)
    writer.write({
        "input":inp,
        "target":trg
    })
    
with open(mbpp,"r",encoding="utf-8") as f:
    for line in f:
        sample=json.loads(line)
        inp=sample["text"]
        trg=sample["code"]
        write_sample(inp,trg)
print("MBPP done")

with open(apps,"r",encoding="utf-8") as f:
    data=json.load(f)
for sample in data:
    inp=sample["prompt"]
    if "canonical_solution" in sample:
        trg=sample["canonical_solution"]
    else:
        continue
    write_sample(inp,trg)
print("APPS done")

with jsonlines.open(ds1000,"r") as reader:
    for sample in reader:
        inp=sample["prompt"]
        trg=sample["reference_code"]
        write_sample(inp,trg)
print("DS1000 done")

file=next(humaneval.glob("*.parquet"))
df=pd.read_parquet(file)
for _,row in df.iterrows():
    inp=row["prompt"]
    trg=row["canonical_solution"]
    write_sample(
        inp,
        trg
    )
print("HumanEval done")

file=next(humanevalplus.glob("*.parquet"))
df=pd.read_parquet(file)
for _,row in df.iterrows():
    inp=row["prompt"]
    trg=row["canonical_solution"]
    write_sample(
        inp,
        trg
    )
print("HumanEval+ done")

writer.close()
print("Saved to",output)