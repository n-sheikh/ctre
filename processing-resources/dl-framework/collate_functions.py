from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import csv
import itertools
import sys
import embedding_functions


def identify_embedding_fns(config_dict):
    feature_embedding_fns = {}
    for feat_fn in [pair.split("-") for pair in config_dict["feature_name_embedding_fn"].split(",")]:
        fn = getattr(embedding_functions, f"{feat_fn[1].strip()}")
        fn_params = {}
        if feat_fn[1] == "llm_token_embedding" or feat_fn[1] == "mirim_llm_token_embedding":
            fn_params["llm_name"] = config_dict["llm_name"]
        feature_embedding_fns[feat_fn[0].strip()] = [fn, fn_params]
    return feature_embedding_fns
add_special_tokens=False

class CNCCollateFn:
    def __init__(self, **kwargs):
        self.connl_folder_path = kwargs["connl_folder_path"]
        self.embedding_functions = identify_embedding_fns(kwargs)

    def __call__(self, batch):
        embeddings = []
        labels = []
        ids = []
        for sample in batch:
            ids.append(sample["id"])
            labels.append(sample["label"])
        for ft_name in self.embedding_functions.keys():
            self.embedding_functions[ft_name][1]["ids"] = ids
            self.embedding_functions[ft_name][1]["connl_folder_path"] = self.connl_folder_path
            self.embedding_functions[ft_name][1]["ft_name"] = ft_name
            embeddings.append(self.embedding_functions[ft_name][0](self.embedding_functions[ft_name][1]))
        return ids, embeddings, torch.LongTensor(labels)


class MiRIMsCollateFn:
    def __init__(self, **kwargs):
        self.connl_folder_path = kwargs["connl_folder_path"]
        self.embedding_functions = identify_embedding_fns(kwargs)

    def __call__(self, batch):
        embeddings = []
        labels = []
        ids = []
        for sample in batch:
            ids.append(sample["id"])
            labels.append(sample["label"])
        for ft_name in self.embedding_functions.keys():
            self.embedding_functions[ft_name][1]["ids"] = ids
            self.embedding_functions[ft_name][1]["connl_folder_path"] = self.connl_folder_path
            self.embedding_functions[ft_name][1]["ft_name"] = ft_name
            embeddings.append(self.embedding_functions[ft_name][0](self.embedding_functions[ft_name][1]))
        return ids, embeddings, torch.LongTensor(labels)


class TransformerCollateFn:
    def __init__(self, **kwargs):
        self.connl_folder_path = kwargs["connl_folder_path"]
        self.embedding_functions = identify_embedding_fns(kwargs)

    def __call__(self, batch):
        embeddings = []
        labels = []
        ids = []
        for sample in batch:
            ids.append(sample["id"])
            labels.append(sample["label"])
        for ft_name in self.embedding_functions.keys():
            self.embedding_functions[ft_name][1]["ids"] = ids
            self.embedding_functions[ft_name][1]["connl_folder_path"] = self.connl_folder_path
            self.embedding_functions[ft_name][1]["ft_name"] = ft_name
            embeddings.append(self.embedding_functions[ft_name][0](self.embedding_functions[ft_name][1]))
        keys = list(self.embedding_functions.keys())
        #[emb#1, emb#2, emb#3]
        #emb#1 [samp#1, samp#2]
        #sample#1 tensor

        for i in range(len(embeddings)):
            if torch.is_tensor(embeddings[i][0]):

                    concatenated_embedding.append(sample_embedding)
            else
                concatenated_embeddings.append(embedding)
        return ids, embeddings, torch.LongTensor(labels)

'''
def connl_file_to_feature_vector(file_path, max_nos_tokens, sep):
    token_sep = []
    if sep:
        token_sep.append(-1)
    with open(file_path) as f:
        csvreader = csv.reader(f, delimiter="\t")
        rows = []
        for row in csvreader:
            features = row[1: len(row) - 1]
            rows.append([int(el) for el in features])
        feature_vector = [-1]
        padding_vector = []
        for i in range(len(rows[0])):
            padding_vector.append(0)
        for i in range(max_nos_tokens):
            if i < len(rows):
                feature_vector = feature_vector + rows[i] + token_sep
            else:
                feature_vector = feature_vector + padding_vector + token_sep
    return feature_vector


class FVConcatCollateFn:
    def __init__(self, **kwargs):
        self.connl_folder_path = kwargs['connl_folder_path']
        self.max_nos_tokens = int(kwargs['max_nos_tokens'])
        self.sep = bool(kwargs['token_sep'])

    def __call__(self, batch):
        sentences = []
        labels = []
        ids = []
        feature_vectors = []
        for sample in batch:
            sentences.append(sample["text"])
            labels.append(sample["label"])
            ids.append(sample["id"])
        for i in range(len(sentences)):
            connl_file_path = f"{self.connl_folder_path}/{ids[i]}.cnnl"
            feature_vectors.append(connl_file_to_feature_vector(connl_file_path, self.max_nos_tokens, self.sep))
        return ids, torch.LongTensor(feature_vectors), torch.LongTensor(labels)
'''