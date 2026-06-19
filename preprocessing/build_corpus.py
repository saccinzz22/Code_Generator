import sys
import json
import jsonlines # type: ignore
from pathlib import Path
import pandas as pd

sys.set_int_max_str_digits(0)

output_file= "tokenizer/all_text.txt"
root = Path("datasets")
mbpp = root / "Code_generation" / "mbpp.jsonl"
apps= root / "Code_generation" / "APPS+.json"
ds1000=root/"DS1000/ds1000.jsonl"
humaneval=root/"humaneval"
humaneval_plus=root/"humaneval+"

def write_text(f,text):
    if text is None:
        return
    text=str(text).strip()
    if len(text)==0:
        return
    f.write(text)
    f.write("\n\n")
with open(output_file, "w", encoding="utf-8") as out:
    with open(mbpp, "r", encoding="utf-8") as f:
        for line in f:
            sample=json.loads(line)
            write_text(out, sample["text"])
            write_text(out, sample["code"])
    print("MBPP done")

    with open(apps, "r", encoding="utf-8") as f:
        data=json.load(f)
    for sample in data:
        write_text(out, sample["prompt"])
        if "canonical_solution" in sample:
            write_text(out, sample["canonical_solution"])
    print("APPS+ done")

    with jsonlines.open(ds1000, "r") as reader:
        for sample in reader:
            write_text(out, sample["prompt"])
            write_text(out, sample["reference_code"])
            write_text(out, sample["code_context"])
    print("DS1000 done")
    
    file=next(humaneval.glob("*.parquet"))
    df=pd.read_parquet(file)
    for _,row in df.iterrows():
        write_text(out, row["prompt"])
        write_text(out, row["canonical_solution"])
    print("HumanEval done")

    file=next(humaneval_plus.glob("*.parquet"))
    df=pd.read_parquet(file)
    for _,row in df.iterrows():
        write_text(out, row["prompt"])
        write_text(out, row["canonical_solution"])
    print("HumanEval+ done")
print("Corpus saved to", output_file)
