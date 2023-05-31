#!/usr/bin/env python
# coding: utf-8

# In[12]:


seed = 1337
# open the text file to inspect it
with open('D:\Desktop\myGPT\text_clean.txt','r', encoding='utf-8') as f:
    text = f.read()


# In[2]:

len(text)


# In[3]:


chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(len(chars))


# In[4]:


stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[a] for a in s] # encoder: takes a string and returns list of integers
decode = lambda l : ''.join([itos[num] for num in l]) #decoder: takes a list of integers and returns the string

# print(encode("Good Morning".lower()))
# print(decode(encode("Good Morning".lower())))


# In[5]:


# encode the entire text into a tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])


# In[6]:


n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# In[8]:


block_size = 8
train_data[:block_size+1]


# In[10]:


x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('for the context ',context, ' we have the target ',target)


# In[35]:


torch.manual_seed(seed)
batch_size = 64
block_size =256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device ',device)
eval_iters = 500
n_embd = 384
learning_rate = 3e-4
n_head = 6
n_layer = 6
dropout = 0.3
max_iters = 4000

# Data Loading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(0,len(data)-block_size,(batch_size,))
    x = torch.stack([data[t:t+block_size] for t in ix])
    y = torch.stack([data[t+1:t+1+block_size] for t in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] =losses.mean()
    m.train()
    return out


# xb, yb = get_batch('train')
# print(xb.shape)
# print(xb)
# print(yb.shape)
# print(yb)


# In[36]:


# Bi-Gram Language Model
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(seed)

class Block(nn.Module):
    """Transformer block, communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        return x

class FeedForward(nn.Module):
    """A simple layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    """one head of self atention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * (C**-0.5) # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class BiGramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head=n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) #(B,T,C)
        pos_embd = self.position_embedding(torch.arange(T,device=device)) #(T,C)
        x = tok_embd + pos_embd
        x = self.blocks(x) #(B, T, C)
        x = self.ln_f(x) #(B, T, C)
        logits = self.lm_head(x) #(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self,idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) #(B,T,C)
            logits = logits[:,-1,:] #(B,C)
            probs = F.softmax(logits,dim=-1) #(B,C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            idx = torch.cat((idx,idx_next),dim=-1) #(B, T+1)
        return idx
    
    
# m = BiGramLanguageModel()
# m = m.to(device)

# # out, loss = m(xb.to(device),yb.to(device))
# # print(out.shape)
# # print(out)
# # print(loss.shape)
# # print(loss)


# # In[37]:


# chars.index('b')


# In[38]:


# inference
# idx = 30*torch.ones((1,1), dtype=torch.long)
# print(decode(m.generate(idx, max_new_tokens=10)[0].tolist()))


# In[47]:


# Train the model
# Pytorch optmization object
# optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
# for iter in range(max_iters):

#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_iters == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()


# # In[56]:

# # inference
# idx = 30*torch.ones((1,1), dtype=torch.long, device=device)
# print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))

# #Save the model
# torch.save(m.state_dict(), 'D:\Desktop\myGPT\model.pt')

# In[ ]:





# In[ ]:




