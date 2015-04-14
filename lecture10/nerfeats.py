#!/usr/bin/env python

"""
A feature extractor for NER
based on the chunking.py script by 2010,2011 Naoaki Okazaki
"""

# Separator of field values.
separator = ' '

# Field names of the input data (created by createfeats.py).
fields = 'w pos cap y'

# Attribute templates.
templates = (
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -1), ('w',  0)),
    (('w',  0), ('w',  1)),
    (('pos', -2), ),
    (('pos', -1), ),
    (('pos',  0), ),
    (('pos',  1), ),
    (('pos',  2), ),
    (('pos', -2), ('pos', -1)),
    (('pos', -1), ('pos',  0)),
    (('pos',  0), ('pos',  1)),
    (('pos',  1), ('pos',  2)),
    (('pos', -2), ('pos', -1), ('pos',  0)),
    (('pos', -1), ('pos',  0), ('pos',  1)),
    (('pos',  0), ('pos',  1), ('pos',  2)),
    (('cap', 0), ),
    (('cap', -1), ),
    (('cap', 1), ),
    (('cap', 0), ('cap', -1)),
    (('cap', 0), ('cap', 1)),
    (('cap', 0), ('cap', -1), ('cap', 1)),
    (('cap', 0), ('pos', 0)),
    (('cap', 0), ('w', 0)),
    (('w', 0), ('pos', 0)),
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
