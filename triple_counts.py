"""
This script should be called with parameters specifying which chunk of wikipedia to process and size of chunk 
(number of articles to process).
Given the chunk number, this script iterates through each wikipedia article in the chunk, dependency parses
each sentence, and accumulates the co-occurence counts between word pairs (adjective-noun and verb-object) and
context words. This requires the stanford coreNLP server to be running locally on port 9000, and requires
access to a file containg the dictionary of words we consider (one word per line). It also requires a compressed
dump of wikipedia in .bz2 format.

Be sure to appropriately specify the file paths, all of which are listed after the command-line arguments are read.
"""
import sys
import os
import numpy as np
from collections import Counter, defaultdict
import pickle
from nltk.parse.corenlp import CoreNLPDependencyParser as dep_parser
import gensim
import pandas as pd
from matplotlib import pyplot as plt
import wikicorpus_sentence as ws

def main():
    # read command-line arguments
    for arg in sys.argv:
        if arg.startswith('--job='):
            job_iter = int(arg.split('--job=')[1])
        if arg.startswith('--num_articles='):
            num_articles = int(arg.split('--num_articles=')[1])

    # SPECIFY THESE FILE PATHS APPROPRIATELY
    path_to_coreNLP_server = 'http://localhost:{}'.format(9000) # specify the port where CoreNLP server is running
    compressed_wiki = '~/enwiki-latest-pages-articles.xml.bz2' # path to the compressed wiki dump
    vocab_file = "~/research/datasets/rw_vocab_no_stopwords.txt" # location of the vocab file (one word (string) per line)
    save_path_triple_an = "triple_counts_an_{}.pkl".format(job) # save path for adjective-noun-context_word triple counts
    save_path_wordpair_an = "wordpair_counts_an_{}.pkl".format(job) # save path for adjective-noun pair counts
    save_path_triple_vo = "triple_counts_vo_{}.pkl".format(job) # save path for verb-object-context_word triple counts
    save_path_wordpair_vo = "wordpair_counts_vo_{}.pkl".format(job) # save path for verb-object pair counts    

    window_size = 5 # radius of context window (but contexts don't cross sentence boundaries)
    
    # connect to CoreNLP server        
    dependency_parser = dep_parser(url=path_to_coreNLP_server)
    
    # instantiate wikipedia iterator
    wiki = ws.WikiCorpusBySentence(compressed_wiki, dictionary={})
    articles = wiki.get_texts()
        
    # create mapping from word (string) to index (int), using vocab file
    vocab = []
    with open(vocab_file,"r") as f:
        for line in f:
            vocab.append(line.strip("\n"))
    vocab_dict = defaultdict(lambda : -1) # this will return index -1 if key not found
    for i, w in enumerate(vocab):
        vocab_dict[w] = i
   
    # initialize objects that will keep track of co-occurence counts for both types of syntactic wordpairs
    triple_counts_an = Counter()
    wordpair_counts_an = Counter()
    triple_counts_vo = Counter()
    wordpair_counts_vo = Counter()
    
    # iterate to the correct chunk
    skip = (job-1)*num_articles
    for i in range(skip): 
        a = next(articles)
    
    # process each article in the chunk    
    for art_num in range(skip,skip+num_articles):
        if art_num % int(num_articles / 100) == 0 and art_num > 0:
            print("Just hit article {} out of {} ({}%)".format(art_num, skip+num_articles, 100*(art_num-skip) / num_articles))
            print("Number of triples: {}, {}".format(len(triple_counts_an),len(triple_counts_vo)))
            
        # save every 500 articles

        if art_num % 500 == 0 and art_num > 0:
            with open(save_path_triple_an, "wb") as f:
                pickle.dump(triple_counts_an,f)
            with open(save_path_wordpair_an, "wb") as f:
                pickle.dump(wordpair_counts_an,f)
            with open(save_path_triple_vo, "wb") as f:
                pickle.dump(triple_counts_vo,f)
            with open(save_path_wordpair_vo, "wb") as f:
                pickle.dump(wordpair_counts_vo,f)
                
        art = next(articles)
        for snt_num,sent in enumerate(art):
            if len(sent) == 0: # skip over empty sentences...
                continue
            for r in range(0,len(sent),150): # process by chunks of 150 words, since the tagger has memory issues if too big
                text_chunk = sent[r:r+150]
            
                try:
                    dep = next(dependency_parser.parse(text_chunk)) # dependency parse the chunk
                except:
                    print("{}".format(len(text_chunk)))
                    continue
                for i in range(len(text_chunk)): # go through each word in the chunk
                    dep_dict = dep.get_by_address(i+1)
                    head = vocab_dict[dep_dict["word"]]
                    if dep_dict["tag"] is None:
                        continue
                    if (dep_dict["tag"][:2] not in ["NN","VB"]) or (head == -1): # if it's not a noun or verb, or not in vocab, skip it
                        continue

                    for ind in dep_dict["deps"]["amod"]: # get indices of all dependent adjectives if head is noun
                        adj = vocab_dict[dep.get_by_address(ind)["word"]]
                        if adj == -1: # if adjective not in dictionary, skip over it
                            continue
                        wordpair_counts_an[(adj,head)] += 1 # increment the adjective-noun wordpair count

                        # get context words within 5 words of noun (excluding the dependent adjective)
                        if ind<i+1: # if adjective occurs before noun
                            for k in range(max(0,i-window_size-1),i):
                                if k+1 != ind: # if the context word is not the same as the adjective, increment counts
                                    context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                    if context_word > -1: # if the context word is in vocab, increment triple count
                                        triple_counts_an[(adj,head,context_word)] += 1
                            for k in range(i+1,min(len(sent),i+1+window_size)):
                                context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                if context_word > -1:
                                    triple_counts_an[(adj,head,context_word)] += 1
                        else: # if adjective occurs after noun
                            for k in range(max(0,i-window_size),i):
                                context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                if context_word > -1:
                                    triple_counts_an[(adj,head,context_word)] += 1
                            for k in range(i+1,min(len(sent),i+2+window_size)):
                                if k+1 != ind:
                                    context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                    if context_word > -1: # if the context word is in vocab, increment triple count
                                        triple_counts_an[(adj,head,context_word)] += 1

                    for ind in dep_dict["deps"]["dobj"]: # get indices of all direct objects if head is verb
                        obj = vocab_dict[dep.get_by_address(ind)["word"]]
                        if obj == -1: # if adjective not in dictionary, skip over it
                            continue
                        wordpair_counts_vo[(head,obj)] += 1 # increment the adjective-noun wordpair count

                        # get context words within 5 words of verb (excluding the dependent object)
                        if ind<i+1: # if object occurs before head 
                            for k in range(max(0,i-window_size-1),i):
                                if k+1 != ind: # if the context word is not the same as the object, increment counts
                                    context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                    if context_word > -1: # if the context word is in vocab, increment triple count
                                        triple_counts_vo[(head,obj,context_word)] += 1
                            for k in range(i+1,min(len(sent),i+1+window_size)):
                                context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                if context_word > -1:
                                    triple_counts_vo[(head,obj,context_word)] += 1
                        else: # if object occurs after head 
                            for k in range(max(0,i-window_size),i):
                                context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                if context_word > -1:
                                    triple_counts_vo[(head,obj,context_word)] += 1
                            for k in range(i+1,min(len(sent),i+2+window_size)):
                                if k+1 != ind:
                                    context_word = vocab_dict[dep.get_by_address(k+1)["word"]]
                                    if context_word > -1: # if the context word is in vocab, increment triple count
                                        triple_counts_vo[(head,obj,context_word)] += 1
    # save the counters
    with open(save_path_triple_an, "wb") as f:
        pickle.dump(triple_counts_an,f)
    with open(save_path_wordpair_an, "wb") as f:
        pickle.dump(wordpair_counts_an,f)
    with open(save_path_triple_vo, "wb") as f:
        pickle.dump(triple_counts_vo,f)
    with open(save_path_wordpair_vo, "wb") as f:
        pickle.dump(wordpair_counts_vo,f)
    
if __name__ == '__main__':
    main()
