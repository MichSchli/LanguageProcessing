#!/bin/bash
#
# PATHS to directories
PROJECTHOME=~/Documents/LanguageProcessing/Exam
CRFSUITE=/usr/local/bin/crfsuite
TEST=$PROJECTHOME/re/dev.input
NERMODEL=$PROJECTSHOME/models/ner.model
POSMODEL=$PROJECTHOME/models/postagger.model

#Preprocessing the datafile
python Preprocessing.py --input $TEST > tmp_files/pre_output.tmp

#Part of Speech tagging the preprocessed datafile
#python PartOfSpeechTagger.py --input tmp_files/pre_output.tmp --load $POSMODEL > tmp_files/pos_output.tmp

#Generating features using CRFSUITE
python $PROJECTHOME/createData.py --sentences tmp_files/pre_output.tmp --pos tmp_files/pos_output.tmp | python nerfeats.py > tmp_files/crf_features.tmp


$CRFSUITE tag -m $NERMODEL tmp_files/crf_features.tmp > tmp_files/ner_output.tmp
