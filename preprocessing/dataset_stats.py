import json
import jsonlines#type:ignore
import pandas as pd
from pathlib import Path
import sys
sys.set_int_max_str_digits(0)

root=Path("datasets")

mbpp=root/"Code_generation"/"mbpp.jsonl"

apps=root/"Code_generation"/"APPS+.json"

ds1000=root/"DS1000"/"ds1000.jsonl"

humaneval=root/"humaneval"

humanevalplus=root/"humaneval+"

print("="*50)

print("MBPP")

print("="*50)

count=0

with open(
    mbpp,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        sample=json.loads(line)

        if "text" in sample and "code" in sample:

            count+=1

print(count)

print()

print("="*50)

print("APPS+")

print("="*50)

count=0

with open(
    apps,
    "r",
    encoding="utf-8"
) as f:

    data=json.load(f)

for sample in data:

    if "prompt" in sample and "canonical_solution" in sample:

        count+=1

print(count)

print()

print("="*50)

print("DS1000")

print("="*50)

count=0

with jsonlines.open(
    ds1000,
    "r"
) as reader:

    for sample in reader:

        if (

            "prompt" in sample

            and

            "reference_code" in sample

        ):

            count+=1

print(count)

print()

print("="*50)

print("HumanEval")

print("="*50)

file=next(
    humaneval.glob("*.parquet")
)

df=pd.read_parquet(file)

count=len(df)

print(count)

print()

print("="*50)

print("HumanEval+")

print("="*50)

file=next(
    humanevalplus.glob("*.parquet")
)

df=pd.read_parquet(file)

count=len(df)

print(count)

print()

print("="*50)

print("TOTAL")

print("="*50)

total=0

with open(
    "preprocessing/final_dataset.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for _ in f:

        total+=1

print(total)