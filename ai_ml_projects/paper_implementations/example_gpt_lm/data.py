import re
from transformers import PreTrainedTokenizerFast

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")

# Register special tokens
special_tokens = {"pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"}
tokenizer.add_special_tokens(special_tokens)

# IDs for convenience
pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

# Read and clean text
with open("text_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Remove lines like ======, then split paragraphs
raw_text = re.sub(r"=+.*?=+", "", raw_text)
raw_text = raw_text.strip().split("\n\n")
raw_text = [t.strip() for t in raw_text if t.strip()]

# Parameters
context_len = 256

# Final datasets
input_ids_dataset = []
labels_dataset = []

for paragraph in raw_text:
    ids = tokenizer.encode(paragraph)  # token IDs list

    # Break into chunks
    for i in range(0, len(ids), context_len - 2):
        chunk = ids[i:i + (context_len - 2)]

        # Add <bos> at start and <eos> at end
        chunk = [bos_id] + chunk + [eos_id]

        # Pad to full context length
        if len(chunk) < context_len:
            chunk += [pad_id] * (context_len - len(chunk))

        # Prepare input and label sequences
        inputs = chunk[:-1]          # all but last token
        labels = chunk[1:]           # all but first token

        # Keep length consistent (context_len-1 here)
        input_ids_dataset.append(inputs)
        labels_dataset.append(labels)
