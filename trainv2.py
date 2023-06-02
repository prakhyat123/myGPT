from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]


import torch
import torch.nn as nn
import tiktoken
import torch.nn.functional as F

#Hyper parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device ', device)
n_embedding = 768

tokenizer = tiktoken.encoding_for_model("gpt-4")

class BiGramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(tokenizer.n_vocab, n_embedding)
        self.lm_head = nn.Linear(n_embedding, tokenizer.n_vocab)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.lm_head(x)
        return x
    
    def generate(self,idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx) #(B,T)
            logits = logits[:, -1, :] #(B, C)
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,)
            idx = torch.cat((idx, idx_next),dim=-1) #(B, T+1)
        return idx

model = BiGramLanguageModel().to(device)

print(tokenizer.decode(model.generate(torch.tensor([tokenizer.encode("My name is Prakhyat Shankesi")], device=device),200)[0].tolist()))