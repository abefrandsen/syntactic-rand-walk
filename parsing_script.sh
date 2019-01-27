#!/bin/sh
#$ -S /bin/bash
#$ -t 1-20
#$ -cwd

# put anaconda python on the path of terminal
export PATH="~/anaconda3/bin:$PATH"

# change into directory containing stanford dependency parser
cd ~/nltk_data/stanford-corenlp-full-2018-02-27/

# start up the parsing server, redirect output and separate from terminal
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -quiet true >/dev/null 2>&1 &

# change to directory that holds the python code for parsing
cd ~/research/tensor_decomp_embedding-master/

# call the python script that iterates through a chunk of wikipedia, does dependency parsing, and accumulates co-occurrence counts
python triple_counts.py --job=$SGE_TASK_ID
