# Anchor
This repository has code for the paper [*High-Precision Model-Agnostic Explanations*](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf).  

An anchor explanation is a rule that sufficiently “anchors” the
prediction locally – such that changes to the rest of the feature
values of the instance do not matter. In other words, for instances on which the anchor holds, the prediction is (almost)
always the same.

At the moment, we support explaining individual predictions for text classifiers or classifiers that act on tables (numpy arrays of numerical or categorical data). If there is enough interest, I can include code and examples for images.

The anchor method is able to explain any black box classifier, with two or more classes. All we require is that the classifier implements a function that takes in raw text or a numpy array and outputs a prediction (integer)

## Installation
The Anchor package is on pypi. Simply run:

    pip install anchor-exp
Or clone the repository and run:

    python setup.py install

If you want to use `AnchorTextExplainer`, you have to run the following:

    python -m spacy download en_core_web_lg

And if you want to use BERT to perturb inputs (recommended), also install transformers:

    pip install torch transformers spacy && python -m spacy download en_core_web_sm


#### Examples
See notebooks folder for tutorials. Note that from version 0.0.1.0, it only works on python 3.

- [Tabular data](https://github.com/marcotcr/anchor/blob/master/notebooks/Anchor%20on%20tabular%20data.ipynb)
- [Text data](https://github.com/marcotcr/anchor/blob/master/notebooks/Anchor%20for%20text.ipynb)

## Citation
[Here](https://homes.cs.washington.edu/~marcotcr/aaai18.bib) is the bibtex if you want to cite this work.
