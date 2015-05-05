__author__ = 'Michael'

from sklearn import svm
import argparse
import Preprocessing
import itertools
import Postprocessing

class RelationClassifier():
    # Fields:
    mode = None
    prediction_classifier = None
    existence_classifier = None

    # Constants for featurization:
    tested_ner = ['PER', 'LOC']

    #Labels:
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

            if self.mode == 'extra_label':
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
            self.existence_classifier.fit(train_data, exist_labels)

            #Construct instances and labels for classification:
            class_instances = [train_data[i] for i in xrange(len(train_data)) if train_labels[i] != 0]
            class_labels = [train_labels[i] for i in xrange(len(train_data)) if train_labels[i] != 0]

            #Fit the prediction classifier with the desired values of C and gamma:
            self.prediction_classifier = svm.SVC(kernel='rbf', C=self.p_C, gamma=self.p_gamma)
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
            return [self.prediction_classifier.predict([data[i]]) if existing[i] else 0 for i in xrange(len(existing))]


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
        if self.mode == 'extra_label':
            matches = [r for r in relations_in_sentence if
                       r['e1_start'] == e1['start'] and r['e1_end'] == e1['end'] and
                       r['e2_start'] == e2['start'] and r['e2_end'] == e2['end']]
            if not matches:
                return 0
            else:
                return self.label_names.index(matches[0]['type'])


'''
Execution:
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contains implementation for two relation extraction strategies.")
    parser.add_argument("--noshell", help="Testing for non-shell environment.", required=False, action='store_true')
    args = parser.parse_args()

    if args.noshell:
        print "Preprocessing..."

        # Get the data:
        sentences, relations, ne, pos = Preprocessing.parse_full_re_file('data/kill+birthplace.baseline')

        print "Training..."

        # Create a test model:
        rc = RelationClassifier('extra_label', [1000, 0.01])

        #Train the model:
        rc.fit_sentences(sentences, ne, pos, relations)

        print "Testing..."

        #Evaluate on train data:
        predictions = rc.predict_sentences(sentences, ne, pos)

        #Postprocessing.print_sentence_relation_list(sentences, predictions)