__author__ = 'Michael'

from sklearn import svm
from sklearn.linear_model import LogisticRegression
import argparse
import Preprocessing
import itertools
import Postprocessing
import Metrics
import numpy as np
import Crossvalidation

'''
Utility methods:
'''
# Constants for featurization:
TESTED_NER = ['PER', 'LOC']

# Labels:
LABEL_NAMES = ['negative', 'kill', 'birthplace']

#Featurize a 4-tuple consisting of sentence and indices for entities:
def featurize(sentence, e1, e2, pos):
    #Sanity check. We can speed stuff up slightly by getting rid of this:
    assert 0 <= e1['start'] <= e1['end']
    assert e1['end'] < len(sentence)
    assert 0 <= e2['start'] <= e2['end']
    assert e2['end'] < len(sentence)

    #Binary variables representing the NE type of the first element of the entity:
    feature = [1.0 if (t in e1['type']) else -1.0 for t in TESTED_NER]
    feature.extend([1.0 if (t in e1['type']) else -1.0 for t in TESTED_NER])

    #Order of the two:
    feature.append(e2['start'] - e1['end'] > 0)

    #Distance:
    feature.append(e2['start'] - e1['end'] if feature[-1] else e1['start'] - e2['end'])

    #Cheat feature. Remove this
    #feature.append(1.0 if 'Oswald' in sentence[e1['start']:e1['end']+1] and 'Kennedy' in sentence[e2['start']:e2['end']+1] else -1.0)

    return feature


#Determine if a set of labels has a relation between two sentences:
def get_match(e1, e2, relations_in_sentence):
    matches = [r for r in relations_in_sentence if
                       r['e1_start'] == e1['start'] and r['e1_end'] == e1['end'] and
                       r['e2_start'] == e2['start'] and r['e2_end'] == e2['end']]
    return matches[0]['type'] if matches else 0

def has_match(e1, e2, relations_in_sentence):
    return get_match(e1,e2, relations_in_sentence) != 0

'''
Relation detection:
'''


class RelationDetector():

    classifier = None
    mode = None

    def __init__(self, mode, params=[1,1]):
        self.mode = mode

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

    def __init__(self, mode, params):
        self.mode = mode

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

    #Fit the relation detector to a dataset of sentences with a corresponding set of accepted relations
    def fit_sentences(self, train_data, relations):
        global_features = []
        global_labels = []
        for i, data in enumerate(train_data):
            sentence, ne, pos = data

            #Get all combinations of named entities that actually have a match:
            ne_combinations = map(list, itertools.product(ne, repeat=2))
            ne_combinations = [c for c in ne_combinations if has_match(n[0], n[1], relations[i])]

            #Featurize the combinations:
            local_features = [featurize(sentence, n[0], n[1], pos) for n in ne_combinations]
            local_labels = [self.get_match(n[0], n[1], relations[i]) for n in ne_combinations]

            global_features.extend(local_features)
            global_labels.extend(local_labels)

        #Train the classifier:
        self.fit(global_features, global_labels)

    def predict_sentences(self, train_data, discovered_relations):
        pass
'''
Legacy:
'''

class RelationClassifierOld():
    # Fields:
    mode = None
    prediction_classifier = None
    existence_classifier = None

    # Constants for featurization:
    tested_ner = ['PER', 'LOC']

    # Labels:
    label_names = ['negative', 'kill', 'birthplace']

    def __init__(self, mode, params):
        self.mode = mode

        self.p_C = params[0]
        self.p_gamma = params[1]

        if self.mode == 'predict_existence':
            self.e_C = params[2]
            self.e_gamma = params[3]

    #Fit the svm to a dataset of sentences
    def fit_sentences(self, sentences, ne, pos, relations):
        global_features = []
        global_labels = []
        for i, sentence in enumerate(sentences):
            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne[i], repeat=2))

            #Featurize the combinations:
            local_features = [self.featurize(sentence, n[0], n[1], pos[i]) for n in ne_combinations]

            local_labels = [self.get_match(n[0], n[1], relations[i]) for n in ne_combinations]

            global_features.extend(local_features)
            global_labels.extend(local_labels)

        #Train the classifier:
        self.fit(global_features, global_labels)


    #Fit an SVM model to a dataset:
    def fit(self, train_data, train_labels):
        #If the n+1 classifier is chosen:
        if self.mode == 'extra_label':
            #Fit the prediction classifier with the desired values of C and gamma:
            self.prediction_classifier = svm.SVC(kernel='rbf', C=self.p_C, gamma=self.p_gamma)
            self.prediction_classifier.fit(train_data, train_labels)

        #If the pre-computation classifier is chosen:
        elif self.mode == 'predict_existence':
            #Construct a binary set of labels corresponding to existence:
            exist_labels = [int(label != 0) for label in train_labels]

            #Fit the existence classifier with the desired values of C and gamma:
            self.existence_classifier = svm.SVC(kernel='rbf', C=self.e_C, gamma=self.e_gamma)
            #self.existence_classifier = LogisticRegression()
            self.existence_classifier.fit(train_data, exist_labels)

            #Construct instances and labels for classification:
            class_instances = [train_data[i] for i in xrange(len(train_data)) if exist_labels[i] != 0]
            class_labels = [train_labels[i] for i in xrange(len(train_data)) if exist_labels[i] != 0]

            #print "Q", sum([c for c in class_labels if c == 1])
            #print "Q", sum([c for c in class_labels if c == 2])

            #Fit the prediction classifier with the desired values of C and gamma:
            self.prediction_classifier = svm.SVC(kernel='rbf', C=self.p_C, gamma=self.p_gamma)
            #self.prediction_classifier = LogisticRegression()
            self.prediction_classifier.fit(class_instances, class_labels)


    def predict_sentences(self, sentences, ne, pos):
        extracted_relations = []
        for i, sentence in enumerate(sentences):
            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne[i], repeat=2))

            #Featurize the combinations:
            features = [self.featurize(sentence, n[0], n[1], pos[i]) for n in ne_combinations]

            #Make a set of sentence predictions:
            predicts = self.predict(features)
            print predicts
            extracted_relations.append([])

            for k in xrange(len(predicts)):
                if predicts[k] != 0:
                    e1 = ne_combinations[k][0]
                    e2 = ne_combinations[k][1]
                    extracted_relations[-1].append(
                        {'type': self.label_names[predicts[k]], 'e1_start': e1['start'], 'e1_end': e1['end'],
                         'e2_start': e2['start'], 'e2_end': e2['end']})

        return extracted_relations

    #Predict a set of labels given that the classifier has already been trained:
    def predict(self, data):
        #If the n+1 classifier is chosen:
        if self.mode == 'extra_label':
            return self.prediction_classifier.predict(data)
        #If the pre-computation classifier is chosen:
        elif self.mode == 'predict_existence':
            existing = self.existence_classifier.predict(data)
            return [self.prediction_classifier.predict([data[i]])[0] if existing[i] else 0 for i in xrange(len(existing))]


    #Featurize a 5-tuple consisting of sentence and indices for entities:
    def featurize(self, sentence, e1, e2, pos):
        #Sanity check. We can speed stuff up slightly by getting rid of this:
        assert 0 <= e1['start'] <= e1['end']
        assert e1['end'] < len(sentence)
        assert 0 <= e2['start'] <= e2['end']
        assert e2['end'] < len(sentence)

        #Binary variables representing the NE type of the first element of the entity:
        feature = [float(t in e1['type']) for t in self.tested_ner]
        feature.extend([float(t in e2['type']) for t in self.tested_ner])

        #Distance between the two entities:
        feature.append(e2['start'] - e1['end'])

        return feature


    def get_match(self, e1, e2, relations_in_sentence):
        #print relations_in_sentence, e1, e2
        matches = [r for r in relations_in_sentence if
                       r['e1_start'] == e1['start'] and r['e1_end'] == e1['end'] and
                       r['e2_start'] == e2['start'] and r['e2_end'] == e2['end']]
        if not matches:
            return 0
        else:
            return self.label_names.index(matches[0]['type'])

    #Internal evaluation used for parameter selection
    def evaluate(self, train_sentences, train_ne, train_pos, train_relations, test_sentences, test_ne, test_pos,
                 test_relations):
        self.fit_sentences(train_sentences, train_ne, train_pos, train_relations)
        results = self.predict_sentences(test_sentences, test_ne, test_pos)

        gold = []
        pred = []
        for i in xrange(len(test_sentences)):
            #Get all combinations of named entities:
            ne_combinations = map(list, itertools.product(ne[i], repeat=2))

            gold.extend([self.get_match(n[0], n[1], test_relations[i]) for n in ne_combinations])

            pred.extend([self.get_match(n[0], n[1], results[i]) for n in ne_combinations])

        return Metrics.precision(pred, gold, 3),Metrics.recall(pred, gold, 3),Metrics.f1(pred, gold, 3)


'''
Execution:
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contains implementation for two relation extraction strategies.")
    parser.add_argument("--noshell", help="Testing for non-shell environment.", required=False, action='store_true')
    args = parser.parse_args()

    if args.noshell:

        print "preprocessing"
        # Get the data:
        sentences, relations, ne, pos = Preprocessing.parse_full_re_file('data/kill+birthplace.data')
        '''
        print "setting up"
        # Create a test model:
        rc = RelationDetector('SVM', [1000, 0.01])

        print "fitting"
        # Train the model:
        rc.fit_sentences(zip(sentences, ne, pos), relations)


        print "predicting"
        rc.predict_sentences(zip(sentences, ne, pos))

        #Evaluate on train data:
        print "evaluating"
        print rc.evaluate_sentences(zip(sentences, ne, pos), relations)
        '''
        Crossvalidation.find_best_svm_params_detector(zip(sentences, ne, pos), relations)

        #Postprocessing.print_sentence_relation_list(sentences, predictions)


        #rc = RelationClassifier('extra_label', [1000, 0.01])
        #print rc.evaluate(sentences, ne, pos, relations,sentences, ne, pos, relations)