import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Configuration
class Config:
    block_size = 32
    n_embd = 768
    n_layer = 2
    n_head = 8
    dropout = 0.1
    learning_rate = 3e-4
    max_grad_norm = 1.0
    epochs = 5   
    batch_size = 32
    max_new_tokens = 50
    temperature = 1.0
    top_k = 10

config = Config()

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, x):
        return self.token_embedding(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, n_embd):
        super().__init__()
        self.positional_embedding = nn.Embedding(block_size, n_embd)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        return self.positional_embedding(pos)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = TokenEmbedding(vocab_size, config.n_embd)
        self.pos_emb = PositionalEmbedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head, config.dropout) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, vocab_size, bias=False)

    def forward(self, idx, mask=None):
        idx = torch.clamp(idx, 0, self.vocab_size - 1)
        token_embeddings = self.token_emb(idx)
        position_embeddings = self.pos_emb(idx)
        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx = torch.clamp(idx, 0, self.vocab_size - 1)
            logits = self(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.clamp(idx_next, 0, self.vocab_size - 1)

            idx = torch.cat((idx, idx_next), dim=1)

            if idx.max().item() >= self.vocab_size:
                raise ValueError(f"Generated index out of range: {idx.max().item()}")

        return idx

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.vocab = sorted(set(text))
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}
        self.data = [self.char2idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.block_size if len(self.data) >= self.block_size else 0

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

def create_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask

def train(model, dataset, config):
    if len(dataset) == 0:
        raise ValueError("Dataset is too small for the block size.")

    loader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(config.epochs):
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, create_mask(x.size(1)).to(device))
            loss = criterion(logits.view(-1, model.token_emb.token_embedding.num_embeddings), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read text from a file
with open('text_data.txt', 'r') as file:
    text = file.read()

dataset = TextDataset(text, config.block_size)
vocab_size = len(dataset.vocab)
model = GPT(vocab_size, config).to(device)

train(model, dataset, config)

input_text = "This is a test"

for ch in input_text:
    if ch not in dataset.char2idx:
        raise ValueError(f"Character '{ch}' not in the vocabulary")

input_indices = [dataset.char2idx[ch] for ch in input_text]
input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)

model.eval()
generated_indices = model.generate(input_tensor, max_new_tokens=config.max_new_tokens, temperature=config.temperature, top_k=config.top_k)
generated_text = ''.join([dataset.idx2char[idx.item()] for idx in generated_indices[0]])

print("Input Text: ", input_text)
print("Generated Text: ", generated_text)
 
