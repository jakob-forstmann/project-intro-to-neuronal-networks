# ConvSeq2Seq in JoeyNMT 
This repo contains my project for the proseminar Introduction to Neuronal Networks and Sequence-To-Sequence Learning at Heidelberg University.
The goal of this project is to implement the [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf) approach porposed by Facebook AI Research in JoeyNMT.

## installing  
```
git clone https://github.com/jakob-forstmann/project-intro-to-neuronal-networks.git 
cd project-intro-to-neuronal-networks

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install JoeyNMT dependencies
cd joeynmt
pip3 install -r requirements.tx
``` 
## download the data 
execute the script `get_wmt14_bpe_conv.sh` located in the folder `joey-nmt/scripts`.
This will download the data, apply preprocessing and learn the BPE. 
More specifically the scripts applies the following:
1. split the downloaded in train, test and validation set 
2. normalize punctuation,remove non printable characters and tokenize the training data using the moses library
3. remove redundant characters,empty lines and drop lines longer than 250 tokens  using the moses library
 
## modified files:
The following files from the original Joey-NMT project were modified or added
1. in the folder joeynmt:
    - attention.py
    - cnn_layers.py
    - decoders.py
    - encoders.py
    - initialization.py
    - model.py 

2. in the test folder:
    - test_cnn_encoder.py
    - test_attention.py 
    - test_model_initialization.py 
