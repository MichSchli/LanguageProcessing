__author__ = 'Michael'

'''
Imports:
'''

import PartOfSpeechTagger as pos
import ModelBuilder

'''
Model evaluation:
'''

def test_model(test_data, test_labels, algorithm, model):
    algorithm.load(model)
    return algorithm.evaluate(test_data, test_labels)


'''
Execution:
'''

if __name__ == '__main__':
    #TODO: Write this class
    pass