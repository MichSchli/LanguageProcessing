__author__ = 'Michael'

'''
Imports:
'''
import LanguageDetector as ld
import Preprocessor
import codecs
import PartOfSpeechTagger as pos

'''
Model construction:
'''

def train_model(train_data, train_labels, algorithm, outfile, params):
    #Construct the model:
    algorithm.fit(train_data, train_labels, *params)

    #Save it:
    algorithm.save(outfile)

'''
Naive Bayes training:
'''

def load_nb_data(file_path):
    sentences = Preprocessor.get_sentence_list(file_path)
    labels = ['EN']*667+['NL']*(1334-667)+['PT']*(2000-1334)
    return sentences, labels

def train_naive_bayes():
    model = ld.NaiveBayes()
    train_data, train_labels = load_nb_data('train.sentences.txt')

    train_model(train_data, train_labels, model, 'models/nb.model')

'''
Structured Perceptron training:
'''

#Note: this is copypasta
def load_sp_data(file_path):
    current_words = []
    current_tags = []

    for line in codecs.open(file_path, encoding='utf-8'):
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

def train_structured_perceptron():
    model = pos.StructuredPerceptron()
    fail = load_sp_data('data/train_dev_data/PT.5000.train')

    train_data = []
    train_labels = []
    for _, (d,l) in enumerate(fail):
        train_data.append(d)
        train_labels.append(l)

    train_model(train_data, train_labels, model, 'models/sp-pt.model', (5, 0.2))


'''
Execution:
'''

if __name__ == '__main__':
    train_structured_perceptron()