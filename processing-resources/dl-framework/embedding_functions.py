import pickle
import csv
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch


def return_col_from_connl_file(sample_id, ft_name, connl_folder_path):
    column_label_file_path = connl_folder_path.strip() + "/column_labels.txt"
    connl_file_path = f"{connl_folder_path}/{sample_id}.cnnl"
    with open(column_label_file_path) as f:
        column_labels = [line.strip() for line in f.readlines()]
    col_id = column_labels.index(ft_name.strip())
    with open(connl_file_path) as f:
        csvreader = csv.reader(f, delimiter="\t")
        column = []
        for row in csvreader:
            column.append(row[col_id])
    return column


def llm_token_embedding(kwargs):
    tokenizer = AutoTokenizer.from_pretrained(kwargs["llm_name"])
    token_embeddings = []
    tokens = []
    for sample_id in kwargs["ids"]:
        tokens.append(return_col_from_connl_file(sample_id, kwargs["ft_name"], kwargs["connl_folder_path"]))
    token_embeddings.append(tokenizer(tokens, padding=True, return_offsets_mapping=True, is_split_into_words=True))
    return token_embeddings


def mirim_llm_token_embedding(kwargs):
    tokenizer = AutoTokenizer.from_pretrained(kwargs["llm_name"])
    token_embeddings = []
    tokens = []
    for sample_id in kwargs["ids"]:
        tokens.append(return_col_from_connl_file(sample_id, kwargs["ft_name"], kwargs["connl_folder_path"]))
    token_embeddings.append(tokenizer(tokens, padding=True, return_offsets_mapping=True, is_split_into_words=True, add_special_tokens=False))
    return token_embeddings


def category_embedding(kwargs):
    with open("/home/nadia/Documents/CLaC-Lab/ctre/processing-resources/dl-framework/resources/category_embeddings.pkl",
              "rb") as f:
        category_embeddings = pickle.load(f)
    category_feature_embedding = category_embeddings[kwargs["ft_name"]]
    embeddings = []
    for sample_id in kwargs["ids"]:
        sample_embedding = []
        column = return_col_from_connl_file(sample_id, kwargs["ft_name"], kwargs["connl_folder_path"])
        for cat in column:
            sample_embedding.append([category_feature_embedding[cat]])
        embeddings.append(torch.LongTensor(sample_embedding))
    return embeddings


def pos_trained_embedding(kwargs):
    with open("/home/nadia/Documents/CLaC-Lab/ctre/processing-resources/dl-framework/resources/trained_pos_emb.pkl",
              "rb") as f:
        trained_pos_embedding = pickle.load(f)
    embeddings = []
    for sample_id in kwargs["ids"]:
        sample_embedding = []
        column = return_col_from_connl_file(sample_id, kwargs["ft_name"], kwargs["connl_folder_path"])
        for cat in column:
            sample_embedding.append(trained_pos_embedding[cat])
        embeddings.append(torch.LongTensor(sample_embedding))
    return embeddings


def binary_embedding(kwargs):
    embeddings = []
    for sample_id in kwargs["ids"]:
        column = return_col_from_connl_file(sample_id, kwargs["ft_name"], kwargs["connl_folder_path"])
        for cat in column:
            embedding = [[float(el.strip())] for el in column]
        embeddings.append(torch.LongTensor(embedding))
    return embeddings


#Proofread this function
def one_hot_embedding(kwargs):
    with open("/home/nadia/Documents/CLaC-Lab/ctre/processing-resources/dl-framework/resources/category_embeddings.pkl",
              "rb") as f:
        category_embeddings = pickle.load(f)
    category_labels = [key for key in category_embeddings[kwargs["ft_name"]].keys()]
    embeddings = []
    for sample_id in kwargs["ids"]:
        sample_embedding = []
        column = return_col_from_connl_file(sample_id, kwargs["ft_name"], kwargs["connl_folder_path"])
        for cat in column:
            token_embedding = [0 for i in range(len(category_labels))]
            index = category_labels.index(cat)
            token_embedding[index] = 1
            sample_embedding.append(token_embedding)
        embeddings.append(torch.LongTensor(sample_embedding))
    return embeddings


def dependency_syntactic_positional_encoding(kwargs):
    embeddings = []
    for sample_id in kwargs["ids"]:
        sample_embedding = [[]]
        root_index = 0
        column = return_col_from_connl_file(sample_id, kwargs["ft_name"], kwargs["connl_folder_path"])
        for el in column:
            el_list = el.split("-")
            if len(el_list) == 1:
                if el_list[0] == 'null':
                    continue
                else:
                    root_index = int(el_list[0])
            else:
                el_list[0] = int(el_list[0])
                el_list[2] = int(el_list[2])
                sample_embedding[0].append(tuple(el_list))
        sample_embedding.append(root_index)
        embeddings.append(sample_embedding)
    return embeddings

