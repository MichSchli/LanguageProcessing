#!/bin/bash
#
# PATHS to directories
PROJECTHOME=~/Documents/LanguageProcessing/Exam
CRFSUITE=/usr/local/bin/crfsuite
TEST=$PROJECTHOME/re/dev.input
NERMODEL=$PROJECTHOME/models/ner.model  #VIRKER IKKE
POSMODEL=$PROJECTHOME/models/postagger.model

#Preprocessing the datafile
python $PROJECTHOME/Preprocessing.py --input $TEST > tmp_files/pre_output.tmp

#Part of Speech tagging the preprocessed datafile
#python $PROJECTHOME/PartOfSpeechTagger.py --input tmp_files/pre_output.tmp --load $POSMODEL > tmp_files/pos_output.tmp

#Turnes a file into conll format so we can dependency parse
python $PROJECTHOME/ParsetoConll.py --sentences tmp_files/pre_output.tmp --pos tmp_files/pos_output.tmp > tmp_files/conll_output.tmp

#Runs MST parser and gives dependency parse
#java -cp mstparser.jar:lib/trove.jar mstparser.DependencyParser test test-file:tmp_files/conll_output.tmp model-name:mst.model output-file:dependency.tmp

#Generating features using CRFSUITE
python $PROJECTHOME/createData.py --sentences tmp_files/pre_output.tmp --pos tmp_files/pos_output.tmp | python nerfeats.py > tmp_files/crf_features.tmp

$CRFSUITE tag -m $NERMODEL tmp_files/crf_features.tmp > tmp_files/ner_output.tmp

python RelationExtraction.py --sentences tmp_files/pre_output.tmp --detector_model models/r_detect.model --classifier_model models/r_class.model --pos tmp_files/pos_output.tmp --ne tmp_files/ner_output.tmp > result.txt
