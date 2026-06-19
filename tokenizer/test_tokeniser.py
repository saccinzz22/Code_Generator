import sentencepiece as spm # type: ignore 

tokenizer=spm.SentencePieceProcessor()
tokenizer.load("coding_assistant.model")

print("Vocabulary Size:")
print(tokenizer.get_piece_size())

text="""
<GENERATE>

Implement binary search.
"""

tokens=tokenizer.encode(text)

print("\nEncoded:")
print(tokens)

print("\nDecoded:")
print(tokenizer.decode(tokens))

print(tokenizer.encode("binary_search"))
print(tokenizer.encode("max_subarray_sum"))
print(tokenizer.encode("customer_transaction_history"))

text="""
<GENERATE>

Given an integer array nums and an integer target,
return indices of the two numbers such that they add
up to target.
"""

tokens=tokenizer.encode(text)

print(len(tokens))
print(tokens)
print(tokenizer.decode(tokens))

for token in tokenizer.encode("binary_search"):

    print(
        token,
        tokenizer.id_to_piece(token)
    )