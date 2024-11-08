# Cochrane-auto
This directory contains all our data, as well as the code used to create our new dataset, for the paper Cochrane-auto: An Aligned Dataset for the Simplification of Biomedical Abstracts.

## Pretrained model
We share the checkpoint for the neural CRF alignment model which we pretrained on Wiki-manual [here](https://drive.google.com/file/d/12FHcrrPdqKgE6R4G7uuTUasuAS9da018/view?usp=sharing).

## Data & code
We share the train/val/test splits of our updated Cochrane corpus under [data/corpus](data/corpus).

The script [load_data.py](load_data.py) contains our code for extracting the sentences and paragraphs from the technical abstracts and lay summaries in this corpus.

The script [alignment.py](alignment.py) contains our code for automatically aligning these sentences using the pretrained alignment model and for computing its performance on a manually annotated subset.

The resulting Cochrane-auto alignments can be found together with our manual alignments under [data/alignments](data/alignments).

The script [preprocessing.py](preprocessing.py) contains our code for constructing the preprocessed Cochrane-auto datasets based on these alignments. It also contains our code for preprocessing the unaligned Cochrane data.

Finally, the resulting sentence-, paragraph- and document-level datasets can be found in the [data](data) directory. 
