"""

This CharGPT (Character Generation Pre-trained Transformer) clearly shows how each component is built and how they
interact by tests cases after each block, intuitive variable names, and data shapes marked in comments, which is a great
teaching material for total beginners. It draws inspiration from Andrej Karpathy’s great tutorial video "Let's build GPT:
from scratch, in code, spelled out" (2022).

Note: Hyperparameters have been intentionally minimized for educational purposes, which may limit bertPretrain performance.
Please scale up key parameters (e.g., embedding dimensions, layer count, and network width) based on available
computational resources to achieve optimal results.

Development Credit: This program was developed by Yijiao Cao (Identifier: 曹一骄1989-ShenzhenMiddleSchool2008-
XidianUniv2012-StonyBrookUniv2015), with all rights reserved. The unique academic identifier follows standard scholarly
disambiguation practices to ensure accurate attribution and distinguish the author within professional communities.
For inquiries, please contact: yijcao@qq.com.

"""


import torch
import torch.nn as nn
from torch.nn import functional as F
import os


# Hyperparameters
batTim = 100  # batch-time or attention-size
batSiz = 32  # batch-size
drpRat = 0.1  # dropout-rate
embDim = 128  # embedding-dimension
lssEst = 8  #  loss-estimation times
numBlc = 10  # number-of-blocks
numHea = 8  # number-of-heads (divider of embDim)
optLrnRat = 3e-3  # optimation-Learning-Rate
trnItr = 500  # training-iteration times
trnItrObs = 50  # training-iteration-observation
trnRat = 0.8  #  training-rate for tokens split
heaSiz = embDim // numHea


# Read text
with open(r'Text.txt', 'r', encoding='utf-8') as file:
  text = file.read()  #  read the file contents
vocabulary = sorted(list(set(text)))  #  get all characters without repetition
vocSiz = len(vocabulary)  #  vocabulary-size tells how many unique tokens are in the text


# Tokenizer
tokTbl = {ch:i for i,ch in enumerate(vocabulary)}  #  token-table
tokenizer = lambda s: [tokTbl[c] for c in s]  #  string  ->  indices
chrTbl = {i:ch for i,ch in enumerate(vocabulary)}  #  character-table
detokenizer = lambda l: ''.join([chrTbl[i] for i in l])  #  indices  ->  string
tokens = torch.tensor(tokenizer(text), dtype=torch.long)  #  characters  ->  indices


# Token Splitting
splTim = int(trnRat * len(tokens))  #  split-timepoint
trnTok = tokens[:splTim]  #  training-tokens
vldTok = tokens[splTim:]  #  validation-tokens


def get_batch(split):
    tok = trnTok if split == 'train' else vldTok  #  token
    strTokIds = torch.randint(len(tok) - batTim, (batSiz,))  #  starting-token-indices (batSiz)
    inpTok = torch.stack([tok[i: i + batTim] for i in strTokIds])  #  input-tokens (batSiz, batTim)
    trgTok = torch.stack([tok[i + 1: i + batTim + 1] for i in strTokIds])  #  target-tokens (batSiz, batTim)
    return inpTok, trgTok


class Embedding(nn.Module):  #  batched-Input-Token-Sequences  ->  batched-Input-Embeddings
    def __init__(self):
        super().__init__()
        self.tokEmbTbl = nn.Embedding(vocSiz, embDim)  #  token-embedding-table
        self.posEmbTbl = nn.Embedding(batTim, embDim)  #  position-embedding-table

    def forward(self, inpTok):  #  input-tokens (batSiz, batTim)
        inpTokEmb = self.tokEmbTbl(inpTok)  #  input-token-embeddings (batSiz, batTim, embDim)
        posEmb = self.posEmbTbl(torch.arange(inpTok.shape[1])).unsqueeze(0)  #  position-embeddings (1, batTim, embDim)
        inpEmb = inpTokEmb + posEmb  #  input-embeddings (batSiz, batTim, embDim)
        return inpEmb  #  input-embeddings (batSiz, batTim, embDim)


class AttHead(nn.Module):  #  attention-head provides raw material for Attention_Multi_Head()'s adjusting each token embedding according to others.
    def __init__(self):  #  number-of-Embeddings, head-Size, lower-Triangular-Size
        super().__init__()
        self.heaSiz = heaSiz  #  Output dimension of raw material for embedding adjustment. A non-trainable constant stored in Attention_Head()
        self.query = nn.Linear(embDim, heaSiz, bias=False)  #  Asking: Who can get my attention?
        self.key = nn.Linear(embDim, heaSiz, bias=False)  #  Answering: Can I get your attention?
        self.value = nn.Linear(embDim, heaSiz, bias=False)  #  Raw material for token embedding adjustment if the answering closely meets the asking.
        self.dropout = nn.Dropout(drpRat)  #  Randomly zero out some elements to create a new pattern, avoiding over-dependence on some specific element, decreasing the risk of over-fitting.


    def forward(self, inpEmb):  #  Batched-Input-Embeddings (batSiz, batTim, embDim)  ->  Weighted-Sub-Mutual-Adjustments (batSiz, batTim, h_size)
        q = self.query(inpEmb)  #  question-Testing (batSiz, batTim, heaSiz)
        k = self.key(inpEmb)  #  question-answering  (batSiz, batTim, heaSiz)
        v = self.value(inpEmb)  #  raw materials for sub mutual adjustments (batSiz, batTim, heaSiz)
        weights = q @ k.transpose(-1, -2) * (self.heaSiz ** -0.5)  #  weights (batSiz, batTim, batTim) for sub mutual adjustments
        mask = torch.tril(torch.ones(inpEmb.shape[1], inpEmb.shape[1], device=inpEmb.device))
        weights = weights.masked_fill(mask == 0, float('-inf'))  #  masking
        weights = F.softmax(weights, dim=-1)  #  sum to 1
        weights = self.dropout(weights)  #  personalization
        return weights @ v  #  weighted-sub-mutual-adjustments (batSiz, batTim, heaSiz)


class AttMulHead(nn.Module):  #  attention-multi-head attention gives each token embedding adjustment according to others.
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttHead() for _ in range(numHea)])  #
        self.adjPrj = nn.Linear(numHea * heaSiz, embDim)  #  adjustment-projector
        self.dropout = nn.Dropout(drpRat)  #  random personalization

    def forward(self, inpEmb):  #  batchedTokenEmbeddings (batSiz, batTim, embDim)
        subMutAdj = torch.cat([h(inpEmb) for h in self.heads], dim=-1)  #  mutual-adjustments (batSiz, batTim, n_head * heaSiz)
        mutAdj = self.adjPrj(subMutAdj)  #  mutual-adjustments (batSiz, batTim, embDim)
        mutAdj = self.dropout(mutAdj)  #  personalization
        return mutAdj  #  (batSiz, batTim, embDim)


class FFN(nn.Module):  #  feed-forward-network
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embDim, 4 * embDim),
            nn.ReLU(),
            nn.Linear(4 * embDim, embDim),
            nn.Dropout(drpRat),
        )

    def forward(self, mutAdj):  #  mutual-adjustments (batSiz, batTim, embDim)
        return self.net(mutAdj)  #  mutual-adjustments (batSiz, batTim, embDim)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attMulHea = AttMulHead()
        self.ffn = FFN()
        self.lyrNrm_1 = nn.LayerNorm(embDim)  #  with learnable beta and gamma
        self.lyrNrm_2 = nn.LayerNorm(embDim)  #  with learnable beta and gamma

    def forward(self, inpEmb):  #  input-embeddings (batSiz, batTim, embDim)
        inpEmb = inpEmb + self.attMulHea(self.lyrNrm_1(inpEmb))  #  mutually adjusted input-embeddings
        inpEmb = inpEmb + self.ffn(self.lyrNrm_2(inpEmb))  #  self adjusted input-embeddings
        return inpEmb


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.seqBlc = nn.ModuleList([Block() for _ in range(numBlc)])  #  The sequential_blocks compounds several block adjustments. Note: parameters of each block are independent, not repetition of one block.
        self.lyrNrm = nn.LayerNorm(embDim)   #  Added final layer norm
        self.vocPrj = nn.Linear(embDim, vocSiz)  #  vocabularyProjection: modDic_1 (batSiz, batTim, embDim)  ->  logits (batSiz, batTim, vocSiz) which are unsoftmaxed pro-probabilities to predict the next token

    def forward(self, inpTok, trgTok = None):  #  input_tokens (batSiz, batTim), target_tokens (batSiz, batTim)
        inpEmb = self.embedding(inpTok)  #  input-embeddings (batSiz, batTim, embDim)
        for blc in self.seqBlc:
            inpEmb = blc(inpEmb)
        logits = self.vocPrj(self.lyrNrm(inpEmb))   #  logits (batSiz, batTim, vocSiz) - the pro-probabilities for next-token-picking

        if trgTok is None:
            loss = None
        else:
            lgt = logits.view(batSiz * batTim, vocSiz)  #  logits (batSiz * batTim, vocSiz)
            trgTok = trgTok.view(batSiz * batTim)  #  (batSiz * batTim)
            loss = F.cross_entropy(lgt, trgTok)  #  average( cross-entropy( softmax(logits), oneHot(target-tokens) ) )

        return logits, loss

    @torch.no_grad()  #  Turn off gradient tracking
    def generate(self, exsTok, numGen):  #  existent-tokens (batSiz, *), number-of-generation (1,)
        for _ in range(numGen):  #  Iteration of updating existing_tokens by appending a new token each time
            lstTok = exsTok[:, -batTim:]  #  last-tokens (batSiz, batTim), the last few tokens
            logits, _ = self(lstTok)  #  (batSiz, batTim, vocSiz)
            logits = logits[:, -1, :]  #  (batSiz, vocSiz), logits at the last timepoint
            logits = F.softmax(logits, dim=-1)  #  (batSiz, vocSiz), next token's probabilities
            nxtTok = torch.multinomial(logits, num_samples=1)  #  (batSiz, 1), newly generated tokens
            exsTok = torch.cat((exsTok, nxtTok), dim=1)  #  (batSiz, * + 1), add to existent-tokens
        return exsTok  #  (batSiz, T + n_new_tok)


@torch.no_grad()  # Turns off gradient-tracking
def estimate_loss(model, dvs):  #  device
    avrLss = {}  # average-Losses of training data and validation data
    model.eval()  # turn off drop-outs and batch-norms
    for split in ['train', 'value']:
        losses = torch.zeros(lssEst)  # losses (lssEst,)
        for i in range(lssEst):
            inpTok, trgTok = get_batch(split=split)
            inpTok = inpTok.to(device)
            trgTok = trgTok.to(device)
            _, loss = model(inpTok, trgTok)
            losses[i] = loss.item()  #  fill in the losses container. item() extracts the scalar value from a single-element tensor
        avrLss[split] = losses.mean()
    model.train()  # re-enables drp/batchNorm
    return avrLss


# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")
gpt = Transformer().to(device)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=optLrnRat)
if os.path.isfile('_parameters'):
    parameters = torch.load('_parameters', map_location=device)
    gpt.load_state_dict(parameters['gpt_state_dict'])
    optimizer.load_state_dict(parameters['optimizer_state_dict'])
    print("Parameters loaded.")
else:
    print("No existing parameters.")

print('\n\na) Text Generation Before Training:')
with torch.no_grad():
    exsTok = torch.zeros((1, 1), dtype=torch.long, device=device)  #  existing_tokens (1, 1)
    genTok = gpt.generate(exsTok=exsTok, numGen=300)[0].tolist()  #  generated-tokens
print(detokenizer(genTok))

print('\n\nb) Training:\n')
for i in range(trnItr):
    # Get batch and move to device
    inpTok, trgTok = get_batch('train')
    inpTok = inpTok.to(device)
    trgTok = trgTok.to(device)

    # Training step
    logits, loss = gpt(inpTok, trgTok)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Periodic evaluation
    if (i + 1) % trnItrObs == 0:
        avrLss = estimate_loss(gpt, device)
        print(f"step = {i+1}: training loss {avrLss['train']:.4f}, validation loss {avrLss['value']:.4f}")

print('\n\nc) Generation After Training:')
genTok = gpt.generate(exsTok=exsTok, numGen=200)[0].cpu().tolist()
print(detokenizer(genTok), '\n')


# Save Parameters
torch.save({'gpt_state_dict': gpt.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, '_parameters')
print("Parameters saved.")
