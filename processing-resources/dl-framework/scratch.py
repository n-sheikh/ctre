import pickle

category_embeddings = {}
# POS Category Embeddings
with open("/home/nadia/Documents/CLaC-Lab/ctre/processing-resources/dl-framework/resources/trained_pos_emb.pkl", "rb") as f:
    trained_pos_emb = pickle.load(f)
    category_pos_emb = {}
    count = 1
    for key in trained_pos_emb.keys():
        category_pos_emb[key] = count
        count = count + 1
category_embeddings["pos"] = category_pos_emb

# Lookup Category Embeddings
category_lookup_labels = ["0", "person", "location", "date"]
category_lookup_embeddings = {}
count = 0
for label in category_lookup_labels:
    category_lookup_embeddings[label] = count
    count = count + 1
category_embeddings["lookup"] = category_lookup_embeddings

# Trigger Category Embeddings
category_trigger_labels = ["0", "implicit_negative", "explicit_negative", "modal"]
category_trigger_embeddings = {}
count = 0
for label in category_trigger_labels:
    category_trigger_embeddings[label] = count
    count = count + 1
category_embeddings["trigger"] = category_trigger_embeddings

# Causality Category Embeddings
category_trigger_labels = ["0", "mcmTrig", "khooTrig"]
category_trigger_embeddings = {}
count = 0
for label in category_trigger_labels:
    category_trigger_embeddings[label] = count
    count = count + 1
category_embeddings["causality_trigger"] = category_trigger_embeddings

with open("/home/nadia/Documents/CLaC-Lab/ctre/processing-resources/dl-framework/resources/category_embeddings.pkl", "wb") as g:
    pickle.dump(category_embeddings, g)
