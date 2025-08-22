import torch
from transformers import PreTrainedTokenizerFast
from language_model import Model  # Import your Model & Decoder definitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")
special_tokens = {"pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"}
tokenizer.add_special_tokens(special_tokens)

pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

# Load trained model
model = Model(vocab_size=len(tokenizer), max_len=256, emb_dim=512, n_heads=8)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

@torch.no_grad()
def generate(prompt, max_new_tokens=50):
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = [bos_id] + input_ids
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    for _ in range(max_new_tokens):
        # Truncate if longer than context
        if input_ids.size(1) > 256:
            input_ids = input_ids[:, -256:]

        # Forward pass
        logits = model(input_ids)  # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]  # last token's logits
        probs = torch.softmax(logits, dim=-1)

        # Sample or greedy
        next_id = torch.argmax(probs, dim=-1).unsqueeze(0)  # greedy
        # next_id = torch.multinomial(probs, num_samples=1)  # sampling

        # Append token
        input_ids = torch.cat([input_ids, next_id], dim=1)
        # print(tokenizer.decode(next_id))
        # Stop at EOS
        if next_id.item() == eos_id:
            break

    # Decode back to string
    tokens = input_ids[0].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text

# Example
output = generate("stars are", max_new_tokens=30)
print(output)
