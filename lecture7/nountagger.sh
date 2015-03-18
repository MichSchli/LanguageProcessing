#!/bin/bash
#
# Simple shell script that assigns a noun to every token
#
## variable that stores location of files, use this to make absolute paths throught the script
SCRIPTDIR=/home/bplank/lp2/lecture7/
### 
# check that required argument is given
if [ -z "$1" ] ; then
    echo "Please specify a file (with one sentence per line)!"
    exit
fi
# store the input file in variable INPUTFILE
INPUTFILE=$1
#
python $SCRIPTDIR/tagger_noun.py $INPUTFILE > $$tmp.tagged
# show content of tagged file
cat $$tmp.tagged

# cleaning up temporary files
rm $$tmp.tagged


