import sys
import codecs
import argparse
import Preprocessing

def extract_data(sentences, poss, separator=' ', labels=None):
    """
    extract token-level data (features) from file that is NER tagged and the corresponding POS file 
    this method should output the format expected by the crfsuite script, i.e., nerfeats.py
    that then looks at a tokens context and produces the respective crfsuite format
    
    # Separator of field values.
    separator = ' '

    # Field names ==> this is what this script produces, needs to be aligned with nerfeats.py
    fields = 'w pos y'
    # alternative (add whatever you wanna add and make sure its aligned with nerfeats.py)
    fields = 'w pos cap y'
    """

    n_sentences = len(sentences)

    for i in xrange(n_sentences):
        if labels is None:
            line_labels = ['PLACEHOLDER']*len(poss[i])
            
        else:
            line_labels = labels[i]
        for word,pos, label in zip(sentences[i],poss[i],line_labels):
            cap="+" if word[0].isupper() else "-"
            hyphen = '+' if '-' in word else '-'
            l = str(len(word))
            suffix = word.lower()[-3:]
            prefix = word.lower()[:3]
            vowels = "".join(sorted([w for w in word.lower() if w in ['a','e','i','o','u','y']]))



            ## todo: output the cap feature and more 
            ## make sure the format you output here is what the nerfeats.py script expects as fields!
            print separator.join([word.lower(), pos, cap, l, hyphen, prefix, suffix, vowels, label])
        # sentence separator
        print ""

        
        
def read_conll_file(file_name):
    """
    read in a file with format:
    word1    tag1
    ...      ...
    wordN    tagN
    
    returns a list of list with (words,tags) tuples for every sentence
    """
    content=[]

    current_words = []
    current_tags = []
    
    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()
        
        if line:
            word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            content.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # if file does not end in newline (it should...), check whether there is an instance in the buffer
    if current_tags != []:
        content.append((current_words, current_tags))
    return content

    
if __name__=="__main__":

    # parse command line options
    parser = argparse.ArgumentParser(description="""Run a structured perceptron""")
    parser.add_argument("--sentences", help="Preprocessed sentence file (CoNLL format)", required=True)
    parser.add_argument("--pos", help="POS file (CoNLL format)", required=True)
    args = parser.parse_args()

    sentences = Preprocessing.parse_processed_sentence_file(args.sentences)
    pos = Preprocessing.parse_processed_sentence_file(args.pos)
    extract_data(sentences, pos)
