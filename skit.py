import torch
import torch.nn as nn
from torch.nn import functional as F

#hiperparametrii
batch_size = 64 #cate secvente INDEPENDENTE se vor procesa in paralel
block_size = 256 #care e lungimea maxima de caractere generata prin predictie probabilistica
max_iter = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#extragem toate caracterele UNICE care apar in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#mapam caracterele la numere
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #ia un string si o transforma intr o lista de numere
decode = lambda l: ''.join([itos[i] for i in l]) #ia o lista de numere, le transforma in stringuri

#taining si teste
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) #primele 90% vor antrena, restul vor fii trecute ca valori
train_data = data[:n]
val_data = data[n:]

#incarcare date
def get_batch(splits):
    #genereaza un batch de date de inputuri x si targeturi y
    data = train_data if splits == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """Un cap pentru atentie"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #inputuri de marime (batch, time-step, canale)
        #outputuri de marime (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) #(B,T,hs)
        q = self.query(x)#(B,T,hs)
        
        #computa scorul atentiei ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**0.5 # (b, t, hs) @ (b, hs, t) -> (b, t, t)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('--inf')) # (b, t, t)
        wei = F.softmax(wei, dim=-1) #(b, t, t)
        wei = self.dropout(wei)
        
        #agregarea valorilor calculate
        v = self.value(x) # (b, t, hs)
        out = wei @ v # (b, t, t) @ (b, t, hs) -> (b, t, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """mai multe capete pentru atentie in paralel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """un layer linear simplu urmat de o non-linearitate"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
        
class Block(nn.Module):
    """Blockul transfomrator"""

    def __init__(self, n_embd, n_head):
        #n_ebd = limiteaza parametrii dimensiunii, n_head = numarul de headuri pe care le vrem
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)  # <-- This now works

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        #idx e vector de tip (b, t) cu indici de context curent
        for _ in range(max_new_tokens):
            #taiem idx pana la ultimul token de block_size
            idx_cond = idx[:, -block_size:]
            #luam predictiile
            logits, loss = self(idx_cond)
            #luam doar valoarea de la ultimul time-step
            logits = logits[:, -1, :] # -> (b, c)
            #aplicam functia de softmax ca sa luam probabilitatile
            probs = F.softmax(logits, dim=-1) # (b, c)
            # exemplu din distibutie
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
            # appendam indexul in secventa curenta
            idx = torch.cat((idx, idx_next), dim=1) # (b, t+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)
#afiseaza numarul de parametrii ai modelului
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    #cream un optimizator folosind PyTorch
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iter):

    #odata la o oarecare perioada de timp, valorile de loss ar trebui evaluate pe seturi de train si val
    if iter % eval_interval == 0 or iter == max_iter -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f }, val loss {losses['val']:.4f }")

        #feeduim cu un batch de date
        xb, yb = get_batch('train')

        #evaluam pierderile
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    #genereaza din model:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))