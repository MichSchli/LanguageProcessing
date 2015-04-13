__author__ = 'Michael'

import argparse
import codecs
from collections import defaultdict as derp

def get_sentences(file_path):
    f = codecs.open(file_path)

    placeholder = [[]]

    for line in f:
        if line.strip() == '':
            placeholder.append([])
        else:
            placeholder[-1].append(line.strip().split('\t'))

    return placeholder

def process_line(line):
    if line == '':
        return ''

    columns = line.split('\t')

    dependant_first = int(columns[0]) > int(columns[6])
    croc = '<' if dependant_first else '>'


    temp = [columns[3],columns[4],croc, columns[7]]

    return temp

def do_stuff(file_path):
    sentences = get_sentences(file_path)

    placeholder = []
    counts = derp(int)


    for sentence in sentences:
        for columns in sentence:
            dependant_first = int(columns[0]) < int(columns[6])
            croc = '<' if dependant_first else '>'

            head = 'ROOT' if columns[6] == '0' else sentence[int(columns[6])-1][3]

            placeholder.append([head,columns[3],croc, columns[7]])

            counts[(head, columns[3], croc, columns[7])] += 1

    for i,p in enumerate(placeholder):
        if p != '':
            p.append(str(counts[tuple(p)]))
            placeholder[i] = '\t'.join(p)


    return placeholder, counts


def get_most_frequent(file_path, n=10):
    f = codecs.open(file_path)

    counts = derp(int)

    for line in f:
        if line.strip() != '':
            line_result = process_line(line.strip())
            counts[tuple(line_result)] += 1

    print len(counts)

    return sorted(counts.iteritems(),key=lambda (k,v): v,reverse=True)[:n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Derp")
    parser.add_argument("file_path", help="The path to the file to be derped.")
    parser.add_argument("--most_frequent", help="Get the n most frequent.", required=False)
    args = parser.parse_args()


    if args.most_frequent:
        print get_most_frequent(args.file_path, n=int(args.most_frequent))

    else:
        placeholder, c = do_stuff(args.file_path)

        for line in placeholder:
            print line

        print len(c)
        print c[('NOUN', 'PRON', '<', 'nmod:poss')]

