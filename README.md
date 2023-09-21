# ConvSeq2Seq in JoeyNMT 
This repo contains my project for the proseminar Introduction to Neuronal Networks and Sequence-To-Sequence Learning at the Heidelberg University.
The goal of this project is to implement the [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf) approach porposed by Facebook AI Research in JoeyNMT.

## repo structure 
folder joey-nmt contains the original JoeyNMT project with the modifications.

The datasets are not included in this repo but they can be download see below.

## download data 
execute the script `get_wmt14_bpe_conv.sh` in the folder `joey-nmt/scripts`
This will download the data, apply preprocessing and learn the BPE 
More specifically the scripts applies the following:
1. split the downloaded in train, test and validation set 
2. normalize punctuation,remove non printable characters and tokenize the training data using the moses library
3. remove redundant characters,empty lines and drop lines longer than 250 tokens  using the moses library
 