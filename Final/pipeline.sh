#!/bin/bash
#
# PATHS to directories
PROJECTHOME=/home/zrb708/Exam/Final
CRFSUITE=/home/bplank/tools/crfsuite-0.12/bin/crfsuite
TEST=$PROJECTHOME/data/raw/dev.input
OUTPUT=$PROJECTHOME/results/relations.out
NERMODEL=$PROJECTHOME/models/ner.model 
POSMODEL=$PROJECTHOME/models/postagger.model
DET_MODEL=$PROJECTHOME/models/relation/r_detect.model
CLASS_MODEL=$PROJECTHOME/models/relation/r_class.model

#Preprocessing the datafile
python $PROJECTHOME/code/Preprocessing.py --input $TEST > tmp/pre_output.tmp

#Part of Speech tagging the preprocessed datafile
python $PROJECTHOME/code/PartOfSpeechTagger.py --input tmp/pre_output.tmp --load $POSMODEL > tmp/pos_output.tmp

#Turns a file into conll format so we can dependency parse
#python $PROJECTHOME/ParsetoConll.py --sentences ner/sentences.data --pos ner/pos/train_no_sent.pos > tmp_files/conll_output.tmp

#Runs MST parser and gives dependency parse
#java -cp mstparser.jar:lib/trove.jar mstparser.DependencyParser test test-file:tmp_files/conll_output.tmp model-name:mst.model output-file:dependency.tmp

#Generating features using CRFSUITE
python $PROJECTHOME/code/createData.py --sentences tmp/pre_output.tmp --pos tmp/pos_output.tmp | python $PROJECTHOME/code/nerfeats.py > tmp/crf_features.tmp

$CRFSUITE tag -m $NERMODEL tmp/crf_features.tmp > tmp/ner_output.tmp

python $PROJECTHOME/code/RelationExtraction.py --sentences tmp/pre_output.tmp --detector_model $DET_MODEL --classifier_model $CLASS_MODEL --pos tmp/pos_output.tmp --ne tmp/ner_output.tmp > $OUTPUT
