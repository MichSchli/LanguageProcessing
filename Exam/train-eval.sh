#!/bin/bash
#
# PATHS to directories
PROJECTHOME=/home/bplank/lp2/lecture10
CRFSUITE=/home/bplank/tools/crfsuite-0.12/bin/crfsuite
# create features
# on train file
echo "create features on train file"
python $PROJECTHOME/createData.py --ner train.ner --pos pos/train.noun | python nerfeats.py > train.crfsuite
# inspect first lines of file
head train.crfsuite
echo ""
echo "create features on test file"
# on test file
## ADD YOUR CODE
echo ">>add your code"

echo "train CRF"
# train and eval model
$CRFSUITE learn -m ner.model train.crfsuite
echo "test CRF"
echo ">>add your code"

