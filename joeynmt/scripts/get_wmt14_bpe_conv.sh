#!/usr/bin/env bash

# adapted from https://github.com/joeynmt/joeynmt/blob/main/scripts/get_iwslt14_bpe.sh
# adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-wmt14en2de.sh 

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPE_TOKENS=40000
MOSES=`pwd`/mosesdecoder

src=en
tgt=de
lang=en-de
data=datasets
prep=WMT-14
tmp=$prep/tmp

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v9.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v9.de-en"
)

prepare_env(){
    git clone https://github.com/moses-smt/mosesdecoder.git
    if [ ! -d "$SCRIPTS" ]; then
        echo "Please set SCRIPTS variable correctly to point to Moses scripts."
        exit
    fi
}

download_data(){ 
    # downloads each required file and unpack them if required
    mkdir -p $data $tmp $prep
    cd ${data} 
    echo "downloading data in directory ${data}"
    for ((i=0;i<${#URLS[@]};++i)); do
        file=${FILES[i]}
        if [ -f $file ]; then
            echo "$file already exists, skipping download"
        else
            url=${URLS[i]}
            wget "$url"
            if [ -f $file ]; then
                echo "$url successfully downloaded."
            else
                echo "$url not successfully downloaded."
                exit -1
            fi
            if [ ${file: -4} == ".tgz" ]; then
                tar zxvf $file
            elif [ ${file: -4} == ".tar" ]; then
                tar xvf $file
            fi
        fi
    done
    cd .. 
}

preprocess_train_data(){
    # normalize punctuation,remove non printable chars 
    # and tokenize each .de(src) and .en(tgt) file from the corpus
    echo "pre-processing train data..."
    for l in $src $tgt; do
        rm $tmp/train.tags.$lang.tok.$l
        for f in "${CORPORA[@]}"; do
            cat $data/$f.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
        done
    done
}

tokenize_test_data(){
    echo "pre-processing test data..."
    for lang in $src $tgt; do
        if [ "$lang" == "$src" ]; then
            target="src"
        else
            target="ref"
        fi
        clean_test_data $lang $target
        perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
        echo ""
    done 
}

clean_test_data(){
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    # remove the tag seq with the attribute id at the beginning and the end of each line
    # convert every ` apostroph to a single apostroph
    grep '<seg id' $data/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
}

split_datasets_in_train_valid(){
    echo "splitting train and valid "
    # every 100th line is put in the validation set 
    for l in $src $tgt; do
        awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
        awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
    done
    TRAIN=$tmp/train.de-en
    BPE_CODE=$prep/code
    rm -f $TRAIN
    for l in $src $tgt; do
        cat $tmp/train.$l >> $TRAIN
    done
}

apply_bpe(){
    echo "learning BPE"
    codes_file="${tmp}/bpe.${merge_ops}"
    cat "${tmp}/train.${src}" "${tmp}/train.${tgt}" > ${tmp}/train.tmp
    python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "${tmp}/train.tmp" -o "${codes_file}"
    rm "${tmp}/train.tmp"

    echo "applying BPE..."
    for l in ${src} ${tgt}; do
        for p in train valid test; do
            python3 -m subword_nmt.apply_bpe -c "${codes_file}" -i "${tmp}/${p}.${l}" -o "${prep}/${p}.bpe.${merge_ops}.${l}"
        done
    done
}

clean_corpus(){
    # remove redundant characters,empty lines and drop lines longer than 250 tokens 
    # ensure that the ratio of the length of the translated sentences is 1.5 
    # taken from http://www2.statmt.org/moses/manual/manual.pdf
    perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
    perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250
}

main(){
    prepare_env
    download_data
    preprocess_train_data
    tokenize_test_data
    split_datasets_in_train_valid
    apply_bpe
    clean_corpus
    #clean up working dir 
    for L in $src $tgt; do
        cp $tmp/bpe.test.$L $prep/test.$L
    done
}
main 