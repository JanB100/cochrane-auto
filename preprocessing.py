import Levenshtein
import math
import pandas as pd

from load_data import load_cochrane_data
from transformers import BartTokenizerFast
from utils import load_pickle_file


def label(complex_sent, simple_sents, threshold=0.92):
    """ 
    Assigns a complex sentence a simplification operation label 
    based on the simple sentences to which it is aligned.
    """
    if simple_sents == []:
        return "delete"
    elif len(simple_sents) > 1:
        return "split"
    elif Levenshtein.ratio(complex_sent, simple_sents[0]) >= threshold:
        return "ignore"
    
    return "rephrase"


def get_merge_idx(labels, sent_ids, cs_alignments, sc_alignments):
    """ 
    Returns the indices of the complex sentences within a given paragraph
    whose label should be replaced with 'merge'. 
    
    labels: original labels of the complex sentences within the paragraph
    sent_ids: ids of the complex sentences within the paragraph
    cs_alignments: new alignments from the sentences in the corresponding complex document (c) 
                   to the sentences in the corresponding simple document (s)
    sc_alignments: original alignments from the sentences in the simple document (s) 
                   to the sentences in the complex document (c)
    """
    merge_idx = []

    for i, (label, sent_id) in enumerate(zip(labels, sent_ids)):
        # Consecutive complex sentences should be labelled 'merge' if:
        if label == "rephrase":
            # (1) one of them was originally labelled 'rephrase', 
            #     because a simple sentence s_j was aligned to it (s -> c),
            #     and it is newly aligned to s_j (c -> s)
            simp_sent_id = cs_alignments[sent_id] - 1

            if simp_sent_id > -1 and sc_alignments[simp_sent_id] - 1 == sent_id:
                idx = [i]

                # (2) the surrounding sentences were labelled 'delete',
                # (3) and are newly aligned to the same s_j (c -> s)
                j = 1
                while i-j > -1 and cs_alignments[sent_id-j]-1 == simp_sent_id \
                               and labels[i-j] == "delete":
                    idx = [i-j] + idx
                    j += 1

                j = 1
                while i+j < len(sent_ids) and cs_alignments[sent_id+j]-1 == simp_sent_id \
                                          and labels[i+j] == "delete":
                    idx = idx + [i+j]
                    j += 1

                if len(idx) > 1:
                    merge_idx.append(idx)

    return merge_idx


def add_merge_labels(df, cs_alignments, sc_alignments):
    """
    Given the part of a sentence-level dataset that belongs to a single paragraph,
    replace the labels of the complex sentences with 'merge' where it is needed.
    """
    sent_ids = [x["sent_id"] for x in df]
    labels = [x["label"] for x in df]

    for idx in get_merge_idx(labels, sent_ids, cs_alignments, sc_alignments):
        ssid =  df[idx[0]]["simp_sent_id"]

        for i in idx[:-1]:
            df[i] |= {"label": "merge", "simp_sent_id": ssid}

        # The last sentence involved in a merge operation is labelled 'none'
        df[idx[-1]] |= {"label": "none", "simp_sent_id": ssid}


def build_sentence_level_dataset(dois, complex, simple, para_ids, sc_alignments, tokenizer=None, 
                                 cs_alignments=None, use_merge_labels=True, threshold=0.5,
                                 max_length=1024, sep=" <s> ", placeholder="<pad> "):
    """
    Build a sentence-level dataset based on the alignments between sentences.
    """
    sent_df = []

    for i, doc in enumerate(complex):
        if len(set(a for a in sc_alignments[i] if a))  / len(doc) < threshold:
            continue

        if tokenizer:
            input = placeholder * len(doc) + sep.join(doc)

            if len(tokenizer(input).input_ids) > max_length:
                continue

        simp_sent_id, len_para = 0, 0
        pair_id = dois[i].split(".")[2]

        for j, sent in enumerate(doc):
            doc_pos = (j+1)/len(doc)
            simple_sents = [simple[i][k] for k, x in enumerate(sc_alignments[i]) if x-1==j]

            sent_df.append({"pair_id": pair_id, "para_id": para_ids[i][j], "sent_id": j,
                            "complex": sent, "label": label(sent, simple_sents),
                            "simple": str(simple_sents), "simp_sent_id": simp_sent_id, 
                            "doc_pos": doc_pos, "doc_quint": math.ceil(doc_pos/0.2), 
                            "doc_len": len(doc)})

            simp_sent_id += len(simple_sents)
            len_para += 1

            if j+1 == len(doc) or para_ids[i][j+1] > para_ids[i][j]:
                if use_merge_labels:
                    add_merge_labels(sent_df[-len_para:], cs_alignments[i], sc_alignments[i])
                len_para = 0

    return pd.DataFrame(sent_df)


def build_higher_level_dataset(sent_df, para_lvl=False, sep=" <s> "):
    """
    Build a paragraph- or document-level dataset given a sentence-level dataset.
    """
    higher_level_df = []
    cols = ["pair_id", "para_id"] if para_lvl else ["pair_id"]

    for ids, df in sent_df.groupby(cols, sort=False):
        row = {col: id_ for col, id_ in zip(cols, ids)}
        row |= {"complex": sep.join(df.complex)}

        if "simple" in sent_df.columns:
            simple = [x for y in df.simple for x in eval(y)]
            row |= {"simple": sep.join(simple)}

        row |= {col: list(df[col]) for col in df if col not in row.keys()}
        higher_level_df.append(row)

    return pd.DataFrame(higher_level_df)


def build_unaligned_datasets(dois, complex, simple, para_ids, tokenizer,
                             only_doc_level=False, max_length=1024, sep=" "):
    """
    Build datasets of complex sentences, paragraphs and documents.
    Only the document-level dataset will contain references.
    """
    doc_df, para_df, sent_df = [], [], []

    for i, doc in enumerate(complex):
        if len(tokenizer(sep.join(doc)).input_ids) > max_length:
            continue

        pair_id = dois[i].split(".")[2]
        doc_df.append({"pair_id": pair_id, "complex": sep.join(doc), \
                       "simple": sep.join(simple[i])})

        if not only_doc_level:
            for j, sent in enumerate(doc):
                sent_df.append({"pair_id": pair_id, "para_id": para_ids[i][j], \
                                "sent_id": j, "complex": sent})

    doc_df = pd.DataFrame(doc_df)
    if only_doc_level:
        return (doc_df,)
    
    sent_df = pd.DataFrame(sent_df)
    para_df = build_higher_level_dataset(sent_df, para_lvl=True, sep=sep)
    return (doc_df, para_df, sent_df)


def build_all_datasets():
    """ 
    Function demonstrating how we built our datasets. 
    """
    for split in ["train", "val", "test"]:
        dois, complex, complex_para_ids, simple, _ = load_cochrane_data(split)
        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base", add_prefix_space=True)

        # Build unaligned datasets
        only_doc_level = split in ["train", "val"]
        dfs = build_unaligned_datasets(dois, complex, simple, complex_para_ids, \
                                       tokenizer, only_doc_level)
        dfs[0].to_csv(f"cochrane_docs_{split}.csv", index=False)
        if not only_doc_level:
            dfs[1].to_csv(f"cochrane_para_{split}.csv", index=False)
            dfs[2].to_csv(f"cochrane_sents_{split}.csv", index=False)

        # Build aligned datasets
        filtered_sc_alignments = load_pickle_file(f"data/alignments/filtered_sc_{split}.pkl")
        cs_alignments = load_pickle_file(f"data/alignments/cs_{split}.pkl")

        sent_df = build_sentence_level_dataset(dois, complex, simple, complex_para_ids, \
                                               filtered_sc_alignments, tokenizer, \
                                               cs_alignments, use_merge_labels=True)
        para_df = build_higher_level_dataset(sent_df, para_lvl=True, sep=" ")
        doc_df = build_higher_level_dataset(sent_df, para_lvl=False, sep=" ")

        sent_df.to_csv(f"cochraneauto_sents_{split}.csv", index=False)
        para_df.to_csv(f"cochraneauto_para_{split}.csv", index=False)
        doc_df.to_csv(f"cochraneauto_docs_{split}.csv", index=False)
