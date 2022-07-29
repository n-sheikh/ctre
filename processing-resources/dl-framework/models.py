import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertConfig, BertModel
import sys
import itertools
import mi_rim
import math
from typing import Tuple
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def align_embeddings(bert_emb, offset_mapping, strategy):
    out = list()
    if strategy == 'first':
        for indx, mapping in enumerate(offset_mapping):
            if mapping[0] == 0:
                out.append(bert_emb[indx])
        return torch.stack(out, dim=0)
    else:
        sub_emb = list()
        offset_mapping.append((-1, -1))
        for indx, mapping in enumerate(offset_mapping):
            if mapping[0] == -1:
                out = out + [torch.mean(torch.stack(sub_emb, dim=0), dim=0).view(1, -1)]
            elif mapping[0] == 0:
                if len(sub_emb) > 0:
                    out = out + [torch.mean(torch.stack(sub_emb, dim=0), dim=0).view(1, -1)]
                    sub_emb = [bert_emb[indx]]
                else:
                    sub_emb = [bert_emb[indx]]
            else:
                sub_emb.append(bert_emb[indx])
        return torch.stack(out, dim=0).squeeze()

'''
Copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
Used with the Transformer 
'''

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class LLM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.pooling = kwargs["pooling"]
        config = AutoConfig.from_pretrained(kwargs["llm"], hidden_dropout_prob=kwargs["llm_hidden_dropout_prob"], attention_probs_dropout=kwargs["llm_attention_probs_dropout_prob"])
        self.llm = AutoModel.from_pretrained(kwargs["llm"], config=config)
        self.w_attn = nn.Linear(AutoConfig.from_pretrained(kwargs["llm"]).hidden_size, 1)
        self.classifier = nn.Linear(
            AutoConfig.from_pretrained(kwargs["llm"]).hidden_size,
            kwargs["classes"])

    def forward(self, X, device):
        in_ids = torch.LongTensor(X[0][0]["input_ids"]).to(device)
        llm_out = self.llm(in_ids)
        clf_in = None
        if self.pooling == 'cls':
            clf_in = llm_out['last_hidden_state'][:, 0, :]
        elif self.pooling == 'attn':
            llm_token_emb = llm_out['last_hidden_state'][0]
            attn_score = torch.softmax(self.w_attn(llm_token_emb), dim=0)
            v = torch.transpose(llm_token_emb, 1, 0)
            clf_in = torch.matmul(v, attn_score).reshape(1, -1)
        elif self.pooling == 'mean':
            llm_token_emb = llm_out['last_hidden_state'][0]
            clf_in = (torch.sum(llm_token_emb, 0) / float(llm_token_emb.size()[0])).reshape(1, -1)
        clf_out = self.classifier(clf_in)
        return clf_out

    @staticmethod
    def generate_hyperparameter_sets(optimization_config_dict, model_config_dict):
        lf = optimization_config_dict['loss_functions'].split(',')
        lr = [float(lr.strip()) for lr in optimization_config_dict['learning_rates'].split(',')]
        bs = [int(bs.strip()) for bs in optimization_config_dict['batch_sizes'].split(',')]
        op = [op.strip() for op in optimization_config_dict['optimizers'].split(',')]
        llm = [llm.strip() for llm in model_config_dict['llm_name'].split(',')]
        hdo = [float(lr.strip()) for lr in model_config_dict['llm_hidden_dropout_prob'].split(',')]
        ado = [float(lr.strip()) for lr in model_config_dict['llm_attention_probs_dropout_prob'].split(',')]
        plg = [op.strip() for op in model_config_dict['pooling'].split(',')]
        cls = [int(cls.strip()) for cls in model_config_dict['classes'].split(',')]
        me = [int(me.strip()) for me in optimization_config_dict['max_epochs'].split(',')]
        hparams = []
        for loss_function, learning_rate, batch_size, optimizer, llm_name, hidden_dropout_prob, attention_probs_dropout_prob, pooling_strategy, classes, max_epochs in \
                itertools.product(lf, lr, bs, op, llm, hdo, ado, plg, cls, me):
            hparams.append({
                'loss_function': loss_function,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'llm': llm_name,
                'llm_hidden_dropout_prob': hidden_dropout_prob,
                'llm_attention_probs_dropout_prob': attention_probs_dropout_prob,
                'pooling': pooling_strategy,
                'classes': classes,
                'max_epochs': max_epochs
            })
        return hparams


class MiRIM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.pooling = kwargs["pooling"]
        config = AutoConfig.from_pretrained(kwargs["llm"], hidden_dropout_prob=kwargs["llm_hidden_dropout_prob"],
                                            attention_probs_dropout=kwargs["llm_attention_probs_dropout_prob"])
        self.llm = AutoModel.from_pretrained(kwargs["llm"], config=config)
        self.mirim = mi_rim.MI_RIM(kwargs["rnn_type"], kwargs["num_mech"], kwargs["num_active"],
                                   kwargs["hidden_size"], kwargs["input_sizes"])
        self.w_attn = nn.Linear(kwargs["hidden_size"] * kwargs["num_mech"], 1)
        self.classifier = nn.Linear(
            kwargs["hidden_size"] * kwargs["num_mech"],
            kwargs["classes"])

    def forward(self, X, device):
        in_ids = torch.LongTensor(X[0][0]["input_ids"]).to(device)
        llm_out = self.llm(in_ids)
        mirim_in = [align_embeddings(llm_out['last_hidden_state'][0], X[0][0]["offset_mapping"][0], "mean")]
        for i in range(1, len(X)):
            mirim_in.append(X[i][0].to(device))
        for x in mirim_in:
            print(x.shape)
        mirim_out,_ = self.mirim(mirim_in)
        #mirim_out,_ = self.mirim([llm_out['last_hidden_state'][0]])
        clf_in = None
        if self.pooling == 'attn':
            mirim_token_emb = mirim_out
            attn_score = torch.softmax(self.w_attn(mirim_token_emb), dim=0)
            v = torch.transpose(mirim_token_emb, 1, 0)
            clf_in = torch.matmul(v, attn_score).reshape(1, -1)
        elif self.pooling == 'mean':
            mirim_token_emb = mirim_out
            clf_in = (torch.sum(mirim_token_emb, 0) / float(mirim_token_emb.size()[0])).reshape(1, -1)
        clf_out = self.classifier(clf_in)
        return clf_out


    @staticmethod
    def generate_hyperparameter_sets(optimization_config_dict, model_config_dict):
        lf = optimization_config_dict['loss_functions'].split(',')
        lr = [float(lr.strip()) for lr in optimization_config_dict['learning_rates'].split(',')]
        bs = [int(bs.strip()) for bs in optimization_config_dict['batch_sizes'].split(',')]
        op = [op.strip() for op in optimization_config_dict['optimizers'].split(',')]
        llm = [llm.strip() for llm in model_config_dict['llm_name'].split(',')]
        hdo = [float(lr.strip()) for lr in model_config_dict['llm_hidden_dropout_prob'].split(',')]
        ado = [float(lr.strip()) for lr in model_config_dict['llm_attention_probs_dropout_prob'].split(',')]
        plg = [op.strip() for op in model_config_dict['pooling'].split(',')]
        rt = [rt.strip() for rt in model_config_dict['rnn_type'].split(',')]
        nm = [int(nm.strip()) for nm in model_config_dict['num_mech'].split(',')]
        na = [int(na.strip()) for na in model_config_dict['num_active'].split(',')]
        hs = [int(hs.strip()) for hs in model_config_dict['hidden_size'].split(',')]
        # Will need to modify this one there is more than one input size
        ins = [[int(x.strip()) for x in ins.split("-")] for ins in model_config_dict['input_sizes'].split(',')]
        cls = [int(cls.strip()) for cls in model_config_dict['classes'].split(',')]
        me = [int(me.strip()) for me in optimization_config_dict['max_epochs'].split(',')]
        hparams = []
        for loss_function, learning_rate, batch_size, optimizer, llm_name, hidden_dropout_prob, attention_probs_dropout_prob, \
            pooling_strategy, rnn_type, num_mech, num_active, hidden_size, input_sizes, classes, max_epochs in \
                itertools.product(lf, lr, bs, op, llm, hdo, ado, plg, rt, nm, na, hs, ins, cls, me):
            hparams.append({
                'loss_function': loss_function,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'llm': llm_name,
                'llm_hidden_dropout_prob': hidden_dropout_prob,
                'llm_attention_probs_dropout_prob': attention_probs_dropout_prob,
                'pooling': pooling_strategy,
                'rnn_type': rnn_type,
                'num_mech': num_mech,
                'num_active': num_active,
                'hidden_size': hidden_size,
                'input_sizes': input_sizes,
                'classes': classes,
                'max_epochs': max_epochs,
            })
        return hparams





