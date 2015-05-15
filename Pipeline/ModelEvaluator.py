__author__ = 'Michael'

'''
Imports:
'''

import PartOfSpeechTagger as pos
import Preprocessor

'''
Model evaluation:
'''

def test_model(test_data, test_labels, algorithm, model):
    algorithm.load(model)
    return algorithm.evaluate(test_data, test_labels)

'''
Language Detection:
'''

def test_structured_perceptron():
    model = pos.StructuredPerceptron()
    fail = Preprocessor.load_sp_data('data/train_dev_data/EN.dev')

    test_data = []
    test_labels = []
    for _, (d,l) in enumerate(fail):
        test_data.append(d)
        test_labels.append(l)

    acc_en = test_model(test_data, test_labels, model, 'models/postagger.model')
    print "Accuracy on english:", acc_en

    model = pos.StructuredPerceptron()
    fail = Preprocessor.load_sp_data('data/train_dev_data/NL.dev')

    test_data = []
    test_labels = []
    for _, (d,l) in enumerate(fail):
        test_data.append(d)
        test_labels.append(l)

    acc_nl = test_model(test_data, test_labels, model, 'models/sp-nl.model')
    print "Accuracy on dutch:", acc_nl

    model = pos.StructuredPerceptron()
    fail = Preprocessor.load_sp_data('data/train_dev_data/PT.dev')

    test_data = []
    test_labels = []
    for _, (d,l) in enumerate(fail):
        test_data.append(d)
        test_labels.append(l)

    acc_pt = test_model(test_data, test_labels, model, 'models/sp-pt.model')

    print "Accuracy on portuguese:", acc_pt
    print "Total accuracy:", (acc_en+acc_nl+acc_pt)/3.0

'''
Execution:
'''

if __name__ == '__main__':
    test_structured_perceptron()