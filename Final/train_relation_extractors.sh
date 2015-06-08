#!/bin/bash
#

#Creates a dependency parse:
python code/ParsetoConll.py --sentences data/gold/train_sentences.gold --pos data/gold/train_sentences.gold > tmp/conll_output.tmp
#Runs MST parser and gives dependency parse:
java -cp mstparser.jar:lib/trove.jar mstparser.DependencyParser test test-file:tmp/conll_output.tmp model-name:mst.model output-file:tmp/train_dependency.tmp

#Trains the system without dependency features:
python code/RelationExtraction.py --noshell --detector_model models/relations/r_detect.model --classifier_model models/relation/r_class.model --train_file data/gold/train.gold

#Trains the system with dependency features:
python code/RelationExtraction.py --noshell --detector_model models/relations/r_detect_dep.model --classifier_model models/relation/r_class_dep.model --train_file data/gold/train.gold --dependencies ner/tmp/train_dependency.tmp