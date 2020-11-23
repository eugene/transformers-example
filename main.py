import torch, sys
import pandas as pd
import numpy as np  
from torch import nn
import torch.nn.functional as F
from model import Transformer, sample_sentence, build_data
from model import colors as c

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate, batch_size, epochs = 1e-3, 256, 512

# loads the `word` column of `unigram_freq.csv` and lower-cases it.
series = pd.read_csv('unigram_freq.csv')['word'].str.lower()[:2**13]
max_len = series.str.len().max().astype(int) # longest word is 13 chars
train_x, train_y, test_x, test_y = build_data(series, min_len = 3, max_len = max_len)
max_index = int(max(train_x.max(), test_x.max()))

args = {
    'emb_dim':        16,            # Embedding vector dimension
    'n_att_heads':    4,             # Number of attention heads for each transformer block
    'n_transformers': 4,             # Depth of the network (nr. of self-attention layers)
    'seq_length':     max_len,       # Sequence length
    'num_tokens':     max_index + 1, # Vocabulary size (highest index found in dataset)
    'device':         device,        # Device: cuda/cpu
    'wide':           False          # Narrow or wide self-attention
}

stats = { 'loss': [], 'perplexity': [] } # we accomulate and save training statistics here
model = Transformer(**args).to(device)
opt   = torch.optim.Adam(lr=learning_rate, params=model.parameters())

for i in range(epochs):
    model.train()
    opt.zero_grad()
    
    # Sample a random batch of size `batch_size` from the train dataset
    idxs = torch.randint(size=(batch_size,), low=0, high=len(train_x))
    
    output, (emb_mean, emb_max) = model(train_x[idxs])
    loss = F.nll_loss(output, train_y[idxs], reduction='mean')
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    loss.backward()
    opt.step()
    
    # Calculate perplexity on the test-set
    model.eval()
    output_test, _ = model(test_x)
    loss_on_test   = F.nll_loss(output_test, test_y, reduction='mean')
    perplexity     = torch.exp(loss_on_test).item()

    # Update the stats and print something.
    stats['loss'].append(loss.item())
    stats['perplexity'].append(perplexity)
    
    sampled  = sample_sentence(model, "z", max_len = max_len, temperature = 0.5)
    
    to_print = [
        f"{c.HEADER}EPOCH %03d"        % i,
        f"{c.OKBLUE}LOSS %4.4f"        % stats['loss'][-1],
        f"{c.OKGREEN}PERPLEXITY %4.4f" % stats['perplexity'][-1],
        f"\t{c.OKCYAN}%s{c.ENDC}"      % sampled
    ]
    print(" ".join(to_print))

# Finally, save everyting:
torch.save({
    'state_dict':   model.state_dict(), 
    'stats':        stats,
    'args':         args,
    'train_x':      train_x,
    'test_x':       test_x
}, f"words.model.pth")