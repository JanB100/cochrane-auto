import json
import nltk


def sent_tokenize_section(sect, para_id):
    """
    Divides a section into sentences using nltk.
    Also determines which paragraph each sentence belongs to, based on newlines.

    sect: a string representing a document section
    para_id: the id of the first paragraph within the section
    """
    sents = nltk.sent_tokenize(sect)
    between_sents, para_ids = [], []
    i, j = 0, 0

    while i < len(sents):
        if sect[j:].startswith(sents[i]):
            para_id += "\n" in between_sents
            para_ids.append(para_id)

            # Correct for splits on e.g. and i.e.
            while sents[i][-4:] in ["e.g.", "i.e."]:
                sents = sents[:i] + [sents[i]+" "+sents[i+1]] + sents[i+2:]

            between_sents = []
            j += len(sents[i])
            i += 1
        else:
            between_sents.append(sect[j])
            j += 1

    return sents, para_ids


def data_to_abstracts(data, type="abstract"):
    """
    Extracts the abstracts from the Cochrane data, dividing them into sentences and paragraphs.

    data: the json file obtained by running the code from Devaraj et al.
    type: whether to extract the technical abstracts or the plain language summaries (pls)
    """
    abstracts, para_ids_per_abstract = [], []

    for x in data:
        sections = x[type]
        if type == "pls" and x["pls_type"] == "long":
            sections = [{"text": x["pls"]}]

        abstract, para_ids = [], []
        for sect in sections:
            initial_para_id = para_ids[-1] + 1 if para_ids else 0
            sents, pids = sent_tokenize_section(sect['text'], initial_para_id)
            abstract += sents
            para_ids += pids
    
        abstracts.append(abstract)
        para_ids_per_abstract.append(para_ids)

    return abstracts, para_ids_per_abstract


def load_cochrane_data(split="train"):
    """ 
    Returns the doi of each systematic review in the Cochrane dataset splits, 
    along with the sentences from the corresponding technical abstracts (complex) 
    and lay summaries (simple), and the ids of the paragraphs they belong to.

    split: whether to load the data from the train/val/test split.
    """
    with open(f"data/corpus/{split}.json") as f:
        data = json.load(f)

    dois = [x["doi"] for x in data]
    complex, complex_para_ids = data_to_abstracts(data, type="abstract")
    simple, simple_para_ids = data_to_abstracts(data, type="pls")
    return dois, complex, complex_para_ids, simple, simple_para_ids
