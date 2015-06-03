__author__ = 'Michael'

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
import argparse
import Preprocessing
import itertools
import Postprocessing
import Metrics
import numpy as np
import Crossvalidation
import sys
from sklearn.externals import joblib

'''
Utility methods:
'''
# Constants for featurization:
TESTED_NER = ['PER', 'LOC']

# Labels:
LABEL_NAMES = ['', 'OrgBased_In', 'Live_In', 'Work_For']


#Featurize a 4-tuple consisting of sentence and indices for entities:
def featurize(sentence, e1, e2, pos):
    #Sanity check. We can speed stuff up slightly by getting rid of this:
    assert 0 <= e1['start'] <= e1['end']
    assert e1['end'] < len(sentence)
    assert 0 <= e2['start'] <= e2['end']
    assert e2['end'] < len(sentence)

    #NE type of the first element of the entities:
    feature = ['e1_type='+e1['type'][2:],'e2_type='+e2['type'][2:]]

    #Order of the two:
    feature.append('order='+str(e2['start'] - e1['end'] > 0))

    #Distance:
    feature.append(str(e2['start'] - e1['end'] if e2['start'] - e1['end'] > 0 else e1['start'] - e2['end']))

    #Words, index and POS of entities:
    for i in xrange(e1['start'], e1['end']):
        feature.append('e1_index_'+str(i-e1['start'])+'='+str(i))
        feature.append('e1_word_'+str(i-e1['start'])+'='+sentence[i])
        feature.append('e1_pos_'+str(i-e1['start'])+'='+pos[i])

    for i in xrange(e2['start'], e2['end']):
        feature.append('e2_index_'+str(i-e2['start'])+'='+str(i))
        feature.append('e2_word_'+str(i-e2['start'])+'='+sentence[i])
        feature.append('e2_pos_'+str(i-e2['start'])+'='+pos[i])

    r = xrange(e1['end']+1, e2['start']) if e2['start'] - e1['end'] > 0 else xrange(e2['end']+1, e1['start'])
    first_entity_end = e1['end']+1 if e2['start'] - e1['end'] > 0 else e2['end']+1
    for j in r:
        feature.append('between_word='+sentence[j])
        feature.append('between_pos='+pos[j])


    return feature


#Determine if a set of labels has a relation between two sentences:
def get_match(e1, e2, relations_in_sentence):
    matches = [r for r in relations_in_sentence if
                       r['e1_start'] == e1['start'] and r['e1_end'] == e1['end'] and
                       r['e2_start'] == e2['start'] and r['e2_end'] == e2['end']]
    return LABEL_NAMES.index(matches[0]['type']) if matches else 0

def has_match(e1, e2, relations_in_sentence):
    return get_match(e1,e2, relations_in_sentence) != 0

'''
Relation detection:
'''


class RelationDetector():

    classifier = None
    mode = None
    feature_hasher = None
    def save(self, filename):
        joblib.dump(self.classifier, filename)

    def load(self, filename):
        self.classifier = joblib.load(filename)

    def __init__(self, mode, params=[1,1]):
        self.mode = mode
        self.feature_hasher = FeatureHasher(input_type='string')

        if mode == 'SVM':
            self.classifier = svm.SVC(kernel='rbf', C=params[0], gamma=params[1])
        elif mode == 'Logistic':
            self.classifier = LogisticRegression()

    #Fit the relation detector to a dataset of sentences
    def fit_sentences(self, train_data, relations):
        global_features = []
        global_labels = []
        for i, data in enumerate(train_data):
            sentence, ne, pos = data
            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne, repeat=2))

            #Featurize the combinations:
            local_features = [featurize(sentence, n[0], n[1], pos) for n in ne_combinations]
            local_labels = [has_match(n[0], n[1], relations[i]) for n in ne_combinations]

            global_features.extend(local_features)
            global_labels.extend(local_labels)

        #Do the featurization:
        global_features = self.feature_hasher.transform(global_features)

        #Train the classifier:
        self.fit(global_features, global_labels)

    def predict_sentences(self, test_data):
        extracted_relations = []
        for i, data in enumerate(test_data):
            sentence, ne, pos = data

            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne, repeat=2))

            #Featurize the combinations:
            features = [featurize(sentence, n[0], n[1], pos) for n in ne_combinations]
            features = self.feature_hasher.transform(features)

            #Make a set of sentence predictions:
            predictions = self.predict(features)
            extracted_relations.append(predictions)

        return extracted_relations


    #Fit the classifier to a dataset:
    def fit(self, data, labels):
        self.classifier.fit(data, labels)

    #Predict the labels corresponding to a dataset:
    def predict(self, data):
        return self.classifier.predict(data)

    #Evaluate on a dataset:
    def evaluate_sentences(self, test_data, test_relations):
        sentence_pred = self.predict_sentences(test_data)
        pred = list(itertools.chain(*sentence_pred))

        gold = []
        for i,data in enumerate(test_data):
            sentence, ne, pos = data

            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne, repeat=2))

            gold.extend([has_match(n[0], n[1], test_relations[i]) for n in ne_combinations])

        return Metrics.precision(pred, gold, 2),Metrics.recall(pred, gold, 2),Metrics.f1(pred, gold, 2)


'''
Relation classification:
'''


class RelationClassifier():

    mode = None
    classifier = None
    feature_hasher = None
    
    def save(self, filename):
        joblib.dump(self.classifier, filename)

    def load(self, filename):
        self.classifier = joblib.load(filename)

    def __init__(self, mode, params):
        self.mode = mode
        self.feature_hasher = FeatureHasher(input_type='string')

        if mode == 'SVM':
            self.classifier = svm.SVC(kernel='rbf', C=params[0], gamma=params[1])
        elif mode == 'Logistic':
            self.classifier = LogisticRegression()

    #Fit the classifier to a dataset:
    def fit(self, data, labels):
        self.classifier.fit(data, labels)

    #Predict the labels corresponding to a dataset:
    def predict(self, data):
        return self.classifier.predict(data)

    #Fit the relation detector to a dataset of sentences with a corresponding set of relations
    def fit_sentences(self, train_data, train_relations):
        global_features = []
        global_labels = []
        for i, data in enumerate(train_data):
            sentence, ne, pos = data

            #Get all combinations of named entities that actually have a match:
            ne_combinations = map(list, itertools.product(ne, repeat=2))
            ne_combinations = [n for n in ne_combinations if has_match(n[0], n[1], train_relations[i])]

            #Featurize the combinations:
            local_features = [featurize(sentence, n[0], n[1], pos) for n in ne_combinations]
            local_labels = [get_match(n[0], n[1], train_relations[i]) for n in ne_combinations]

            global_features.extend(local_features)
            global_labels.extend(local_labels)

        #Do the featurization:
        global_features = self.feature_hasher.transform(global_features)

        #Train the classifier:
        self.fit(global_features, global_labels)

    #Construct predictions corresponding to a set of data and a set of discovered relations:
    def predict_sentences(self, test_data, discovered_relations, output_dictionary=False):
        extracted_relations = []
        for i, data in enumerate(test_data):
            sentence, ne, pos = data

            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne, repeat=2))

            #Featurize the combinations:
            features = [featurize(sentence, n[0], n[1], pos) for n in ne_combinations]

            #Do the featurization:
            features = self.feature_hasher.transform(features)

            #Make a set of sentence predictions:
            predictions = self.predict(features)

            #Keep only the ones with discovered relations (this is a somewhat slow way to do it, but the code is short):
            predictions = [predictions[j]*discovered_relations[i][j] for j in xrange(len(ne_combinations))]

            #Output the predictions on dictionary format:
            if output_dictionary:
                extracted_relations.append([])
                for k in xrange(len(predictions)):
                    if predictions[k] != 0:
                        e1 = ne_combinations[k][0]
                        e2 = ne_combinations[k][1]
                        extracted_relations[-1].append(
                        {'type': LABEL_NAMES[predictions[k]], 'e1_start': e1['start'], 'e1_end': e1['end'],
                         'e2_start': e2['start'], 'e2_end': e2['end']})
            else:
                extracted_relations.append(predictions)

        return extracted_relations

    #Evaluate on a dataset:
    def evaluate_sentences(self, test_data, test_relations):

        gold = []
        detects = []
        for i,data in enumerate(test_data):
            sentence, ne, pos = data

            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne, repeat=2))

            disc = [get_match(n[0], n[1], test_relations[i]) for n in ne_combinations]

            gold.extend(disc)
            detects.append([g != 0 for g in disc])

        sentence_pred = self.predict_sentences(test_data,detects)
        pred = list(itertools.chain(*sentence_pred))

        return Metrics.precision(pred, gold, 2),Metrics.recall(pred, gold, 2),Metrics.f1(pred, gold, 2)


'''
Execution:
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contains implementation for two relation extraction strategies.")
    parser.add_argument("--noshell", help="Testing for non-shell environment.", required=False, action='store_true')
    parser.add_argument("--sentences", help="Read a preprocessed sentence file.", required=False)
    parser.add_argument("--pos", help="Read a POS file.", required=False)
    parser.add_argument("--ne", help="Read an NER file.", required=False)
    parser.add_argument("--detector_model", help="A model for relation detection", required=False)
    parser.add_argument("--classifier_model", help="A model for relation detection", required=False)
    args = parser.parse_args()

    if args.noshell:

        print >> sys.stderr, "preprocessing"
        # Get the data:
        sentences, relations, ne, pos = Preprocessing.parse_full_re_file('re/train.gold')
        test_sentences, test_relations, test_ne, test_pos = Preprocessing.parse_full_re_file('re/dev.gold')

        Crossvalidation.find_best_svm_params_detector(zip(sentences, ne, pos), relations)

        print >> sys.stderr, "setting up"
        # Create a test model:
        rc = RelationDetector('SVM', [1000, 0.01])

        print >> sys.stderr, "fitting"
        # Train the model:
        rc.fit_sentences(zip(sentences, ne, pos), relations)

        print >> sys.stderr, "predict relations..."
        pred = rc.predict_sentences(zip(test_sentences, test_ne, test_pos))

        print >> sys.stderr, "set up classifier"
        rcl = RelationClassifier('SVM', [1000, 0.01])

        print >> sys.stderr, "training classifier..."
        rcl.fit_sentences(zip(sentences, ne, pos), relations)

        print >> sys.stderr, "classifying"
        predictions = rcl.predict_sentences(zip(test_sentences, test_ne, test_pos), pred, output_dictionary=True)

        print >> sys.stderr, "evaluating"
        print rcl.evaluate_sentences(zip(test_sentences, test_ne, test_pos), test_relations)

    else:
        if args.sentences and args.pos and args.ne and args.detector_model and args.classifier_model:
            # Load in the two models:
            rc = RelationDetector('SVM', [1000, 0.01])
            rc.load(args.detector_model)
            rcl = RelationClassifier('SVM', [1000, 0.01])
            rcl.load(args.classifier_model)

            sentences = Preprocessing.parse_processed_sentence_file(args.sentences)
            pos = Preprocessing.parse_processed_sentence_file(args.pos)
            ne_plain = Preprocessing.parse_processed_sentence_file(args.ne)
            ne = [Preprocessing.process_named_entities(n) for n in ne_plain]

            print >> sys.stderr, "predict relations..."
            pred = rc.predict_sentences(zip(sentences, ne, pos))

            print >> sys.stderr, "classifying"
            predictions = rcl.predict_sentences(zip(sentences, ne, pos), pred, output_dictionary=True)
            Postprocessing.print_sentence_pos_ne_relation_list(sentences, pos, ne_plain, predictions)

        else:
            print >> sys.stderr, 'Missing input'
