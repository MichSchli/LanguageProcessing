#!/bin/bash
#
# PATHS to directories
PROJECTHOME=~/Documents/LanguageProcessing/Exam
CRFSUITE=/usr/local/bin/crfsuite
TEST=$PROJECTHOME/re/dev.input
NERMODEL=$PROJECTSHOME/models/ner.model
POSMODEL=$PROJECTHOME/models/postagger.model

python Preprocessing.py --input $TEST > pre_output.tmp

python PartOfSpeechTagger.py --input pre_output.tmp --load $NERMODEL
