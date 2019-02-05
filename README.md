# Syntactic rand-walk
This repository contains a python implementation of the syntactic rand-walk model introduced by Abraham Frandsen and Rong Ge in their paper [Understanding Composition of Word Embeddings via Tensor Decomposition](https://arxiv.org/abs/1902.00613). The parameters of the model -- the word embeddings themselves as well as a tensor -- are syntax-aware in the sense that they are trained using data that are pre-processed with a syntactic dependency parser. Compositionality refers to the role of the tensor, which can be used to non-linearly combine syntactically-related pairs of word embeddings.  While the training algorithm is the centerpiece of this repository, we will walk through all elements of the code implementation, from data pre-processing, to the optimization algorithm, to evaluation tasks.

Apart from the python packages that the code relies on (notably TensorFlow, NLTK, gensim, NumPy, etc.), this implementation also depends on software from the Stanford CoreNLP project as well as a data dump of English wikipedia.

In order to make this code work on your machine, be sure to change all of the filepaths referenced in the code appropriately.

## Pre-processing
To train the model, we need to estimate co-occurrence counts between pairs and triples of words from a large corpus of natural language text. Here, we use a dump of English wikipedia: download the file 'enwiki-latest-pages-articles.xml.bz2' from [the wikimedia dumps page](https://dumps.wikimedia.org/enwiki/latest/), but don't uncompress the file. 

We also parse the corpus to identify syntactically related word pairs (e.g. adjective-noun and verb-object pairs) and the words that occur in the context of such pairs. We utilize the neural-network-based dependency parser in the [Stanford CoreNLP software library](https://stanfordnlp.github.io/CoreNLP/). After downloading the software packages, we run the CoreNLP server (instructions [here](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)) and access the parser with python via the NLTK python library. 


### Parsing and co-occurrence counts
In order to parse large amounts of text, we break up wikipedia into many chunks of articles, and split up the work of parsing and computing co-occurrence counts over a cluster of machines using the [Oracle Gride Engine](https://en.wikipedia.org/wiki/Oracle_Grid_Engine) system for batch job submission. The file 'parsing_script.sh' performs the batch job submission, first starting up the CoreNLP server, and then invoking the python script for parsing and co-occurrence computation. 

The file 'triple_counts.py' contains the python code to parse a chunk of wikipedia articles and accumulate various co-occurence counts for adjective-noun and verb-object pairs. These counts are needed for the training algorithm. The file 'wikicorpus_sentence.py' contains small but important re-writes of certain funcions in gensim. These re-writes allow us to iterate through each wikipedia sentence-by-sentence, so that we can parse each sentence individually (the intended use case of the dependency parser). 

### Aggregating the counts
After running the batch job, we have potentially many python pickle files containing co-occurrence counts for the chunks of wikipedia articles processed. It is necessary to aggregate these counts, e.g. produce one file that contains the counts for every adjective-noun-context_word triple, etc. These counts can potentially take up a lot of memory, so it is recommended to use a machine with plenty of RAM when combining all of the counts. After aggregating the triple counts, write them to a text file where each line has the following structure:
(index of adjective (verb)) (index of noun (object)) (index of context word) (occurrence count). For example, the first few lines of the file might be as follows:
```
727 2111 9799 1
12417 385 1330 1
795 434 2086 1
13714 188 3999 2
```
The first line above means that the adjective-noun pair (or verb-object pair) corresponding to words 727 and 2111 occurs in the same context as word 9799 just once in the entire corpus. 

Finally, it is recommended to shuffle the lines of the triple counts files, so that the training algorithm draws random batches of co-occurrence counts. This can be done on linux from the terminal using the 'shuf' command. Note also that the training algorithm can be optionally supplied with the number of total triple counts, so that it can report what percentage of the data have been processed. To get the number of triple counts, just count the lines of the file, which on linux can be done from the terminal using 'wc -l'. 


## Training the model
We train the model using the adam optimizer in the TensorFlow framework. The file 'train_from_triplecounts.py' contains the main training function. There are many parameters to this function and many ways to customize the training procedure. However, the primary inputs are as follows:
1. wordpair-context_word triple co-occurrence count file in the format described above. Pass the filepath to the script via the argument '--counts_file'.
2. pre-computed word embeddings in a text file, where line i contains a space-separated list of floats that give the coordinates for the word embedding corresponding to word i (must have the same indexing as used in the triple co-occurrence counts). Pass the filepath to the script via the argument '--vector_file'.
3. initialized model parameters. This is optional if you want to resume training on a partially-trained model. The file containing these parameters should be a NumPy archive file in exactly the same format that 'train_from_triplecounts.py' saves the model parameters. Pass the filepath to the script via the argument '--init_file'. 
4. filepath to the location where you want to save the learned file parameters (including intermediate results). Pass this via the argument '--save_path'. 
5. the number of epochs to train, where each epoch corresponds to a complete pass over the data. Pass this via the argument '--epochs'.

The script 'train_from_triplecounts.py' will run the training algorithm on the specified inputs, and will save the learned model parameters frequently, so that the script can be terminated without losing everything (and can be resumed later without too much difficulty). You can also have it save the value of the loss function over time in order to monitor the progress of the training.

## Using and evaluating the model
The first and most basic use of the trained syntactic rand-walk model is to obtain an embedding for a syntactically-related pair of words. The jupyter notebook 'model_demo.ipynb' shows how to do this, and gives examples of similarity queries for various adjective-noun phrases.

Our paper also applied the model to a phrase similarity task. The jupyter notebook 'phrase_similarity.ipynb' demonstrates how we performed this evaluation task. 
