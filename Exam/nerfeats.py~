#!/usr/bin/env python

"""
A feature extractor for NER
based on the chunking.py script by 2010,2011 Naoaki Okazaki
"""

# Separator of field values.
separator = ' '

# Field names of the input data (created by createfeats.py).
fields = 'w pos cap l hyphen prefix suffix vowels y'

# Attribute templates.
templates = (
    (('w', -3), ),
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),           #Looking at words
    (('w',  1), ),
    (('w',  2), ),
    (('w', -3), ),
    (('w', -1), ('w',  0)),      #Bigrams
    (('w',  0), ('w',  1)),
    (('pos', -3), ),
    (('pos', -2), ),
    (('pos', -1), ),             #Part of Speech
    (('pos',  0), ),
    (('pos',  1), ),
    (('pos',  2), ),
    (('pos', -3), ),
    (('pos', -2), ('pos', -1)),
    (('pos', -1), ('pos',  0)),
    (('pos',  0), ('pos',  1)),            #POS bigrams
    (('pos',  1), ('pos',  2)),
    (('pos', -2), ('pos', -1), ('pos',  0)),
    (('pos', -1), ('pos',  0), ('pos',  1)),    #POS trigrams
    (('pos',  0), ('pos',  1), ('pos',  2)),
    (('cap', 0), ),
    (('cap', -1), ),                           #Capital letter
    (('cap', 1), ),
    (('cap', -1), ('pos', 0)),
    (('cap', 0), ('pos', 0)),              #Capital letter + POS
    (('cap', -1), ('w', 0)),
    (('cap', 0), ('w', 0)),             #Capital letter + word
    (('cap', 0), ('cap', -1)),
    (('cap', 0), ('cap', 1)),           #Capital bigrams
    (('cap', 0), ('cap', -1), ('cap', 1)),    #Capital trigram
    (('w', -1), ('pos', 0)),
    (('w', 0), ('pos', 0)),              #Word + POS
    (('w', 1), ('pos', 0)),             
    (('hyphen', 0), ),
    (('hyphen', 1), ),               #Hyphens
    (('hyphen', -1), ),
    (('l', 0), ),                   #How long is this?
    (('prefix', 0),),
    (('suffix', 0),),
    (('prefix', 1),),               #Affixes
    (('suffix', 1),),
    (('prefix', -1),),
    (('suffix', -1),),
    (('prefix', -1), ('pos', 0)),
    (('prefix', 0), ('pos', 0)),
    (('suffix', -1), ('pos', 0)),            #Affixes + POS
    (('suffix', 0), ('pos', 0)),       
    (('vowels', 0),('l', 0)),                     #Vowels + length of word
    (('vowels', 0),),                            #Vowels
    #add more features (remember to define them in fields) or feature combinations
    )


import crfutils

def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
	# Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature

if __name__ == '__main__':
    crfutils.main(feature_extractor, fields=fields, sep=separator)
