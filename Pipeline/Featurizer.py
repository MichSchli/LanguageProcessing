__author__ = 'Michael'

'''
Imports:
'''
import codecs

'''
Featurization for Naive Bayes:
'''

#Get the default feature extraction for nb: Trigrams
def get_default_nb_extraction_method():
    #Read trigrams from file:
    feat = codecs.open('nb_features.txt')
    trigrams = [f[:3] for f in feat]

    #Construct a function which builds a feature vector:
    return lambda x: [int(t in x) for t in trigrams]

'''
Meta-logical methods:
'''

#Extract features from a sentence based on an extraction method:
def feature_extract_sentence(sentence, extraction_method):
    return extraction_method(sentence)

#Extract features from a text based on an extraction method:
def feature_extract_text(text, extraction_method=None):
    #If we have no extraction method, get the default:
    if extraction_method is None:
        extraction_method = get_default_nb_extraction_method()

    #Construct a feature matrix:
    feature_matrix = [extraction_method(sentence) for sentence in text]
    return feature_matrix

'''
Testing playground:
'''

if __name__ == '__main__':
    print feature_extract_text(['Right, he said.', 'I am going to test this. So much. Wow. Testing. t i. aefeaefaith.'])
