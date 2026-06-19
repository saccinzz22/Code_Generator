def count_examples(filepath="preprocessing/final_dataset.jsonl"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # sum(1 for _ in f) is a fast, memory-efficient way to count lines
            total_examples = sum(1 for _ in f)
        
        print(f"Total examples in '{filepath}': {total_examples}")
        return total_examples
        
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return 0

# Run the function
count_examples()