__author__ = 'Michael'

'''
Imports:
'''
import codecs
import argparse

'''
I/O:
'''
#This is complete shit:
def load_nb_data(file_path):
    sentences = get_sentence_list(file_path)
    labels = ['EN']*667+['NL']*(1334-667)+['PT']*(2000-1334)
    return sentences, labels


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



def get_sentence_list(file_path):
    whitespace = ['\n']
    return [s.strip() for s in codecs.open(file_path) if s not in whitespace]

'''
Testing playground:
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocesses a file")
    parser.add_argument("file_path", help="The path to the file to be preprocessed.")
    args = parser.parse_args()

    for sentence in get_sentence_list(args.file_path):
        print sentence