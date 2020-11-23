# Based on TRANSFORMERS FROM SCRATCH
# http://peterbloem.nl/blog/transformers

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.device          = kwargs['device']
        self.num_tokens = nt = kwargs['num_tokens']
        self.seq_length = sl = kwargs['seq_length']
        self.emb_dim    = ed = kwargs['emb_dim']
        self.n_transformers  = nr = kwargs['n_transformers']
        self.token_embedding = nn.Embedding(embedding_dim=ed, num_embeddings=nt)
        self.pos_embedding   = nn.Embedding(embedding_dim=ed, num_embeddings=sl)

        trans_args = {
            'emb':        self.emb_dim, 
            'heads':      kwargs['n_att_heads'], 
            'mask':       True, 
            'wide':       kwargs['wide']
        }

        trans_blocks = [TransformerBlock(**trans_args) for _ in range(nr)]
        self.tblocks = nn.Sequential(*trans_blocks)
        self.toprobs = nn.Linear(ed, nt)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        
        batch = x.shape[0] # batch size

        # Encode ASCII indexes to embedding vectors of dimension `emb_dim`
        x = self.token_embedding(x) # [batch x seq_length x emb_dim]
        
        # Create a range of possible positions 0..seq_length-1
        pos_range = torch.arange(self.seq_length, device=self.device)

        # Encode the range of positions to vectors
        pos = self.pos_embedding(pos_range)                    # [ seq_len x emb_dim ]
        pos = pos.unsqueeze(0)                                 # [ 1 x seq_len x emb_dim ]
        pos = pos.expand(batch, self.seq_length, self.emb_dim) # [ batch x seq_len x emb_dim ]
        x   = x + pos                                          # [ batch x seq_len x emb_dim ]

        # Output of the last transformer block 
        x = self.tblocks(x) # [batch x seq_len x emb_dim]

        # Extract embeddings by collapsing `seq_len` dimension
        # see https://stackoverflow.com/questions/59030907
        emb_mean, emb_max = x.mean(dim=1), x.max(dim=1)[0]
        
        x = x.view(batch * self.seq_length, self.emb_dim)   # [ batch*seq_len,  emb_dim ]
        x = self.toprobs(x)                                 # [ batch*seq_len,  num_tokens ]
        x = x.view(batch, self.num_tokens, self.seq_length) # [ batch, num_tokens, seq_len ]

        return F.log_softmax(x, dim=1), (emb_mean, emb_max)

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0, wide=True):
        super().__init__()

        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
                    else SelfAttentionNarrow(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class SelfAttentionWide(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

def sample_categorical(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    return dist.Categorical(p).sample()

def sample_sentence(model, query, max_len = 140, temperature=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(max_len - len(query)):
        query_ = torch.zeros(max_len).to(torch.long)
        query_[:len(query)] = torch.from_numpy(np.frombuffer(str.encode(query), np.uint8).copy())
        output, _     = model(query_.unsqueeze(0).to(device))
        next_char_idx = sample_categorical(output[0, :, len(query) - 1], temperature) #0.5
        if next_char_idx <= 1:
            # query += "*"
            break
        query += str(chr(max(32, next_char_idx)))
    
    return query

# Takes in a pandas Series object and returns pytorch tensors
# for the train and test sets
def build_data(series, min_len = 3, max_len = 140):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # all sequences between `min_len` and `max_len`
    series = series[(series.str.len() > min_len) & (series.str.len() < max_len)] 

    # Preprocess strings into tensors of char ascii indexes
    inputs  = torch.zeros((len(series), max_len)).to(torch.long).to(device)
    targets = torch.zeros((len(series), max_len)).to(torch.long).to(device)

    for i, word in enumerate(series):
        try: word.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError: continue
        inputs[i,  0:len(word)]   = torch.from_numpy(np.frombuffer(str.encode(word), np.uint8).copy())
        targets[i, 0:len(word)-1] = torch.from_numpy(np.frombuffer(str.encode(word[1:]), np.uint8).copy())
        targets[i, len(word)-1]   = 1  # <EOS> token

    # Split into train and test dataset
    combined = torch.stack([inputs, targets], dim=1)
    train_size = int(0.8 * len(combined))
    test_size = len(combined) - train_size
    train_ds, test_ds = torch.utils.data.random_split(combined, [train_size, test_size])

    train_x, train_y = combined[train_ds.indices][:, 0, :], combined[train_ds.indices][:, 1, :]
    test_x, test_y   = combined[test_ds.indices][:, 0, :],  combined[test_ds.indices][:, 1, :]

    return train_x, train_y, test_x, test_y

# Nice colors for the terminal
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'