#!/bin/bash
#
# PATHS to directories
PROJECTHOME=~/Documents/LanguageProcessing/Exam
CRFSUITE=/usr/local/bin/crfsuite

# inspect first lines of file
#head train.crfsuite
echo ""
echo "create features on test file"
# on test file
## ADD YOUR CODE
python $PROJECTHOME/createData.py | python nerfeats.py > test.crfsuite


echo "train CRF"
# train and eval model
#$CRFSUITE learn -m models/ner.model test.crfsuite 
echo "test CRF"
$CRFSUITE tag -m models/ner.model test.crfsuite 
