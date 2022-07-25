import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertConfig, BertModel
import sys
import itertools
import mi_rim


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
        in_ids = torch.LongTensor(X["input_ids"]).to(device)
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
                                   kwargs["hidden_size"], [768])
        self.w_attn = nn.Linear(kwargs["hidden_size"] * kwargs["num_mech"], 1)
        self.classifier = nn.Linear(
            kwargs["hidden_size"] * kwargs["num_mech"],
            kwargs["classes"])

    def forward(self, X, device):
        in_ids = torch.LongTensor(X["input_ids"]).to(device)
        llm_out = self.llm(in_ids)
        mirim_out,_ = self.mirim([llm_out['last_hidden_state'][0]])
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
        ins = [[int(ins.strip()) for ins in model_config_dict['input_sizes'].split(',')]]
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


