import argparse
import codecs
from collections import defaultdict, Counter
import cPickle
import numpy as np
import sys
import networkx as nx
import Preprocessing
import Metrics
import Postprocessing
import itertools

np.set_printoptions(precision=4)


def memoize(f):
    """
    helper function to be used as decorator to memoize features
    :param f:
    :return:
    """
    memo = {}
    def helper(*args):
        key = tuple(args[1:])
        try:
            return memo[key]
        except KeyError:
            memo[key] = f(*args)
            return memo[key]
    return helper



class StructuredPerceptron(object):
    """
    implements a structured perceptron as described in Collins 2002
    """

    def __init__(self):
        """
        initialize model
        :return:
        """
        self.feature_weights = defaultdict(float)
        self.tags = set()

        self.START = "__START__"
        self.END = "__END__"


    def fit_sentences(self, train_data, train_labels, iterations=5, learning_rate=0.2):
        """
        read in a CoNLL file, extract emission features iterate over instances to train weight vector
        :param file_name:
        :return:
        """
        averaged_weights = Counter()

        for iteration in xrange(iterations):
            correct = 0
            total = 0.0
            sys.stderr.write('iteration %s\n************\n' % (iteration+1))

            for i, words in enumerate(train_data):
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

                for tag in train_labels[i]:
                    self.tags.add(tag)

                # get prediction
                prediction = self.predict(words)

                # derive global features
                global_gold_features = self.get_global_features(words, train_labels[i])
                global_prediction_features = self.get_global_features(words, prediction)

                # update weight vector
                for fid, count in global_gold_features.iteritems():
                    self.feature_weights[fid] += learning_rate * count
                for fid, count in global_prediction_features.iteritems():
                    self.feature_weights[fid] -= learning_rate * count

                # compute training accuracy for this iteration
                correct += sum([1 for (predicted, gold) in zip(prediction, train_labels[i]) if predicted == gold])
                total += len(train_labels[i])

            sys.stderr.write('\n\t%s features\n' % (len(self.feature_weights)))
            averaged_weights.update(self.feature_weights)
            sys.stderr.write('\tTraining accuracy: %.4f\n\n' % (correct/total))

        self.feature_weights = averaged_weights


    def get_global_features(self, words, tags):
        """
        count how often each feature fired for the whole sentence
        :param words:
        :param tags:
        :return:
        """
        feature_counts = Counter()

        for i, (word, tag) in enumerate(zip(words, tags)):
            previous_tag = self.START if i == 0 else tags[i-1]
            feature_counts.update(self.get_features(word, tag, previous_tag))

        return feature_counts


    @memoize
    def get_features(self, word, tag, previous_tag):
        """
        get all features that can be derived from the word and tags
        :param word:
        :param tag:
        :param previous_tag:
        :return:
        """
        word_lower = word.lower()
        prefix = word_lower[:3]
        suffix = word_lower[-3:]
        vowels = [l for l in word_lower if l in ['a','e','i','o','u']]

        features = [
                    'TAG_%s' % (tag),                                           # current tag
                    'TAG_BIGRAM_%s_%s' % (previous_tag, tag),                   # tag bigrams
                    'WORD+TAG_%s_%s' % (word, tag),                             # word-tag combination
                    'WORD+TAG_BIGRAM_%s_%s_%s' % (word_lower,tag,previous_tag), #word-tag bigram combination
                    'WORD_LOWER+TAG_%s_%s' % (word_lower, tag),                 # word-tag combination (lowercase)
                    'UPPER_%s_%s' % (word[0].isupper(), tag),                   # word starts with uppercase letter
                    'DASH_%s_%s' % ('-' in word, tag),                          # word contains a dash
                    'PREFIX+TAG_%s_%s' % (prefix, tag),                         # prefix and tag
                    'SUFFIX+TAG_%s_%s' % (suffix, tag),                         # suffix and tag
                    'LAST_LETTER_%s_%s' % (word_lower[-1], tag),                # last letter of the word
                    'FIRST_LETTER_%s_%s' % (word_lower[0], tag),                # first letter of the word
                    'LENGTH_%s_%s' %(str(len(word)), tag),                      # length of the word
                    'VOWELS_%s_%s' %(vowels, tag),                              # the vowels used in the word
                    'N_OF_VOWELS_%s_%s' %(str(len(vowels)), tag),               # the number of vowels in the word
        ]

        return features


    def predict_sentences(self, sentences):
        for sentence in sentences:
            yield self.predict(sentence)

    def predict(self, words):
        """
        use Viterbi to predict best sequence
        :param words:
        :param word_features:
        :return:
        """
        number_of_words = len(words)
        number_of_states = len(self.tags)
        tags = list(self.tags)

        V = nx.DiGraph()

        # initialize
        for j, tag in enumerate(tags):
            features = self.get_features(words[0], tag, self.START)
            feature_weights = sum((self.feature_weights[x] for x in features))

            V.add_edge(self.START, "%s_0" % (tags[j]), weight=-feature_weights)

        # iterate
        for i in xrange(1, number_of_words):
            for j, tag in enumerate(tags):
                for k, previous_tag in enumerate(tags):
                    features = self.get_features(words[i], tag, previous_tag)
                    feature_weights = sum((self.feature_weights[x] for x in features))

                    V.add_edge("%s_%s" % (tags[k], i-1), "%s_%s" % (tags[j], i), weight=-feature_weights)

        # add END node
        for j, tag in enumerate(tags):
            V.add_edge("%s_%s" % (tags[j], number_of_words-1), self.END, weight=1.0)

        # find shortest path
        predecessors, edge_weights = nx.bellman_ford(V, self.START)

        current = self.END
        best_path = []
        while current != self.START:
            best_path.append(predecessors[current])
            current = predecessors[current]

        return [node.split('_')[0] for node in best_path[::-1][1:]]


    def evaluate_sentences(self, test_data, test_labels, output=False):
        """
        compute accuracy on a test file
        :param file_name:
        :param output:
        :return:
        """
        correct = 0
        total = 0.0
        if output:
            sys.stderr.write('\nTesting\n')
            sys.stderr.write('*******\n')

        for i, (words) in enumerate(test_data):
            if output:
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

            # get prediction
            prediction = self.predict(words)

            if output:
                print >> sys.stderr, "\t%s" % " ".join(words)
                print >> sys.stderr, "GOLD:\t%s" % " ".join(test_labels[i])
                print >> sys.stderr, "PRED.:\t%s" % " ".join(prediction)
                print >> sys.stderr, ""

            correct += sum([1 for (predicted, gold) in zip(prediction, test_labels[i]) if predicted == gold])
            total += len(test_labels[i])
        if output:
            print >> sys.stderr, "\nTest accuracy on %s items: %.4f" % (i+1, correct/total)

        return [correct/total]

    def read_conll_file(self, file_name):
        """
        read in a file with format:
        word1    tag1
        ...      ...
        wordN    tagN

        Sentences MUST be separated by newlines!

        :param file_name: file to read in
        :return: generator of instances ((list of  words, list of tags) pairs)
        """
        current_words = []
        current_tags = []

        for line in codecs.open(file_name, encoding='utf-8'):
            line = line.strip()

            if line:
                word, tag = line.split('\t')
                current_words.append(word)
                current_tags.append(tag)

            else:
                yield (current_words, current_tags)
                current_words = []
                current_tags = []

        # if file does not end in newline (it should...), check whether there is an instance in the buffer
        if current_tags != []:
            yield (current_words, current_tags)


    def save(self, file_name):
        """
        save model
        :param file_name:
        :return:
        """
        print >> sys.stderr, "saving model...",
        model = {'tags': list(self.tags), 'weights': dict(self.feature_weights)}
        cPickle.dump(model, open(file_name, "wb"))
        print >> sys.stderr, "done"


    def load(self, file_name):
        """
        load model from JSON file
        :param file_name:
        :return:
        """
        f = open(file_name, 'r+')
        model = cPickle.load(f)
        self.tags = set(model['tags'])
        self.feature_weights = defaultdict(float, model['weights'])


# if script is run from command line, automatically execute the following
if __name__ == "__main__":


    # parse command line options
    parser = argparse.ArgumentParser(description="""Run a structured perceptron""")
    parser.add_argument("--validate", help="validate implementation using crossvalidation", action="store_true")
    parser.add_argument("--load", help="load model from file", required=False)
    parser.add_argument("--noshell", required=False, action="store_true")
    parser.add_argument("--input", required=False, help="Specify the input file")
    args = parser.parse_args()

    if args.noshell:

        if not args.validate:
            #Read in the sentences:
            sentences,pos_gold = Preprocessing.parse_sentence_pos_file('pos/train.pos')

            #Load in the trained model:
            print "Training structured perceptron..."
            sp = StructuredPerceptron()
            sp.fit_sentences(sentences, pos_gold, iterations=10, learning_rate=0.2)

            print "Saving model..."
            sp.save('models/postagger.model')


        else:
            sp = StructuredPerceptron()
            sp.load('models/postagger.model')

            print >> sys.stderr, "Evaluating..."

            #Read in the sentences:
            sentences,gold = Preprocessing.parse_sentence_pos_file('pos/train.pos')

            #Tag the sentences:
            sentence_pred = sp.predict_sentences(sentences)
            pred = list(itertools.chain(*sentence_pred))

            #Output nicely:
            print "Train:"
            print "---------------------------"
            print "Precision:",Metrics.precision(pred, gold, 2)
            print "Recall:", Metrics.recall(pred, gold, 2)
            print "F1:",Metrics.f1(pred, gold, 2)
            print "==========================="

            #Read in the sentences:
            sentences,_,_gold = Preprocessing.parse_full_re_file('re/dev.gold')

            #Tag the sentences:
            sentence_pred = sp.predict_sentences(sentences)
            pred = list(itertools.chain(*sentence_pred))

            #Output nicely:
            print "Development:"
            print "---------------------------"
            print "Precision:",Metrics.precision(pred, gold, 2)
            print "Recall:", Metrics.recall(pred, gold, 2)
            print "F1:",Metrics.f1(pred, gold, 2)
            print "==========================="

            #Read in the sentences:
            sentences,_,_gold = Preprocessing.parse_full_re_file('re/test.gold')

            #Tag the sentences:
            sentence_pred = sp.predict_sentences(sentences)
            pred = list(itertools.chain(*sentence_pred))

            #Output nicely:
            print "Test:"
            print "---------------------------"
            print "Precision:",Metrics.precision(pred, gold, 2)
            print "Recall:", Metrics.recall(pred, gold, 2)
            print "F1:",Metrics.f1(pred, gold, 2)
            print "==========================="


    else:

        sp = StructuredPerceptron()
        if args.load and args.input:
            sp.load(args.load)
            sentences = Preprocessing.parse_processed_sentence_file(args.input)
            print >> sys.stderr, "POS-tagging..."
            predictions = sp.predict_sentences(sentences)
            Postprocessing.print_sentence_list(predictions)



    #print tagged

    '''
    # create new model
    sp = StructuredPerceptron()

    if args.load:
        sp.load(args.load)

    if args.train:
        sp.fit(args.train, iterations=args.iterations, learning_rate=args.learning_rate)

    if args.save:
        sp.save(args.save)

    # check whether to show predictions
    if args.test:
        sp.evaluate(args.test, output=args.output)
    '''
