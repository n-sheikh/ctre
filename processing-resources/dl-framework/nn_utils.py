import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, emb_dim,vocab_size,initialize_emb,word_to_ix):
        super(Embedding, self).__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim)
        if initialize_emb:
            inv_dic = {v: k for k, v in word_to_ix.items()}
            for key in initialize_emb.keys():
                if key in word_to_ix:
                    ind = word_to_ix[key]
                    self.embedding.weight.data[ind].copy_(torch.from_numpy(initialize_emb[key]))