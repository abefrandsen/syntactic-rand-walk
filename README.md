# Syntactic rand-walk
This repository implements the syntactic rand-walk model for syntax-aware compositional word embeddings. The parameters of the model -- the word embeddings themselves as well as a tensor -- are syntax-aware in the sense that they are trained using data that are pre-processed with a syntactic dependency parser. Compositionality refers to the role of the tensor, which can be used to non-linearly combine syntactically-related pairs of word embeddings.  While the training algorithm is the centerpiece of this repository, we will walk through all elements of the code implementation, from data pre-processing, to the optimization algorithm, to evaluation tasks.

# Pre-processing

## Parsing through wikipedia

# Training the model
