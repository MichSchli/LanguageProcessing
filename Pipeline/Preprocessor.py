__author__ = 'Michael'

'''
Imports:
'''
import codecs


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
    print get_sentence_list('alice.txt')