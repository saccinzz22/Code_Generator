import sentencepiece as spm # type: ignore

spm.SentencePieceTrainer.train(
input="tokenizer/all_text.txt",
model_prefix="coding_assistant",
vocab_size=24000,
model_type="bpe",
character_coverage=1.0,
pad_id=0,
unk_id=1,
bos_id=2,
eos_id=3,
max_sentence_length=16384,
train_extremely_large_corpus=True
)
print("Tokenizer trained successfully.")