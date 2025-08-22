# from datasets import load_dataset
# from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# dataset = dataset.select(range(20000))  # first 20k samples

# # Write text to a file (tokenizers library trains from files)
# with open("text_data.txt", "w", encoding="utf-8") as f:
#     for item in dataset:
#         f.write(item["text"].replace("\n", " ") + "\n")

# # Create a BPE tokenizer
# tokenizer = Tokenizer(models.BPE())
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# # Train BPE tokenizer
# trainer = trainers.BpeTrainer(vocab_size=16000, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
# tokenizer.train(["text_data.txt"], trainer)

# # Save tokenizer
# tokenizer.save("bpe_tokenizer.json")
