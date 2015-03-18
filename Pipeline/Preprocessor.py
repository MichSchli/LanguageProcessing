__author__ = 'Michael'

'''
Imports:
'''
import codecs
import argparse

'''
I/O:
'''


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

    print get_sentence_list(args.file_path)