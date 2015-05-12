#!/bin/bash
#
# PATHS to directories
PROJECTHOME=~/Documents/LanguageProcessing/Exam
CRFSUITE=/usr/local/bin/crfsuite
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
python $PROJECTHOME/createData.py --ner test.ner --pos pos/test.noun | python nerfeats.py > test.crfsuite


echo "train CRF"
# train and eval model
$CRFSUITE learn -m ner.model train.crfsuite
echo "test CRF"
$CRFSUITE tag -qt -m ner.model test.crfsuite 
