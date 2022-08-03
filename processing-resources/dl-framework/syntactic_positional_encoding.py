import torch.nn as nn
import torch
import numpy as np
import networkx as nx
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class QueryKeyProduct(torch.nn.Module):
    def __init__(self,d_in,d_model):
        super(QueryKeyProduct, self).__init__()
        self.query_layer=nn.Linear(d_in,d_model)
        self.key_layer=nn.Linear(d_in,d_model)
    def forward(self,input):
        query=self.query_layer(input)
        key=self.key_layer(input).transpose(1,0)
        return torch.matmul(query,key)


class SyntacticTransformer(torch.nn.Module):
    #An impelimenaiton of Transformer with Syntactic Position Encoding
    #https://aclanthology.org/2021.ranlp-1.172.pdf
    #By: Parsa Bagherzadeh
    #Date: March 2022
    #CLaC Lab
    def __init__(self, d_model,d_type_emb,dependency_types,d_ff):
        super(SyntacticTransformer, self).__init__()
        self.type2idx={'none':0}
        self.d_model=d_model
        for dep_type in dependency_types:
            if dep_type not in self.type2idx:
                self.type2idx[dep_type]=len(self.type2idx)
        self.type_emb=nn.Embedding(len(self.type2idx),d_type_emb)
        self.lstm_path_encoder=nn.LSTM(input_size=d_type_emb,hidden_size=d_model,batch_first=True)
        self.positional_encoding=PositionalEncoding(d_model,0.1,max_len=200)
        self.query_key_absolute=QueryKeyProduct(d_model,d_model)
        self.query_key_type=QueryKeyProduct(d_type_emb,d_type_emb)
        self.value_layer=nn.Linear(d_model,d_model)
        self.position_wise_ff=nn.Sequential(nn.Linear(d_model,d_ff),nn.ReLU(),nn.Linear(d_ff,d_model))

    def get_paths(self,input_len,dependency_triples,root_idx):
        dep_graph = nx.DiGraph()
        paths = list()
        for triple in dependency_triples:
            dep_graph.add_edge(triple[0], triple[2], label=triple[1])
        for t in range(input_len):
            if dep_graph.has_node(t):
                sp = nx.shortest_path(dep_graph, source=root_idx, target=t)
                path = [(sp[i], sp[i + 1],) for i in range(0, len(sp) - 1)]
                path_labels = [self.type2idx[dep_graph.get_edge_data(path[i][0], path[i][1])['label']] for i in range(0, len(sp) - 1)]
                paths.append(path_labels)
            else:
                paths.append([])
        return paths

    def make_batch(self,paths):
        max_len=max([len(path) for path in paths])
        batched=[path+[self.type2idx['none']]*(max_len-len(path)) for path in paths]
        return batched

    def forward(self, input, dependency_triples, root_idx, device):
        ##Importatnt-> I named the variable according to the paper
        ## P: the sequences of dependency types from root to a certain token
        ## S: The path encoding obtained by LSTM
        P=self.get_paths(len(input),dependency_triples,root_idx)
        P=torch.tensor(self.make_batch(P),dtype=torch.long).to(device)
        P=self.type_emb(P)
        S=self.lstm_path_encoder(P)[1][0].view(-1,1,self.d_model).squeeze()
        X=self.positional_encoding(input.unsqueeze(0)).squeeze(0)
        attention_score=torch.softmax(self.query_key_absolute(X)+self.query_key_absolute(S),dim=1)
        X=input+torch.matmul(attention_score,self.value_layer(X))
        X=X+self.position_wise_ff(X)
        return X

#dep_types=['subj','det','obj','subjpass','cop','nmod']

#model=SyntacticTransformer(d_model=768,d_type_emb=64,dependency_types=dep_types,d_ff=512)
#output=model(torch.randn(4,768),[(1,'subj',0),(1,'obj',3),(3,'det',2)],1)

