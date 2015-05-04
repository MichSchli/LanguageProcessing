__author__ = 'Michael'
import codecs

def read_relation(line):
    return line.strip().split('\t')

def parse_re_file(filename):
    sentences = []
    sentence = ([], [])
    parsed_sentence = False
    for line in codecs.open(filename):
        #If there is a newline:
        if not line.strip():
            if not parsed_sentence:
                parsed_sentence = True
            else:
                sentences.append(sentence)
                sentence = ([], [])
                parsed_sentence = False
        #If there is something:
        else:
            if not parsed_sentence:
                sentence[0].append(line.strip().split('\t'))
            else:
                sentence[1].append(read_relation(line))

    if not sentence == ([],[]):
        sentences.append(sentence)

    return sentences

def parse_full_re_file(filename):
    sentences = [[]]
    relations = [[]]
    nes = [[]]
    poss = [[]]

    parsed_sentence = False
    doner = False
    for line in codecs.open(filename):
        #If there is a newline:
        if not line.strip():
            if not parsed_sentence:
                parsed_sentence = True
            else:
                doner = True
                parsed_sentence = False
        #If there is something:
        else:
            if not parsed_sentence:
                if doner:
                    sentences.append([])
                    relations.append([])
                    nes.append([])
                    poss.append([])
                    doner = False

                l = line.strip().split('\t')
                sentences[-1].append(l[1])
                poss[-1].append(l[2])
                nes[-1].append(l[3])
            else:
                relations[-1].append(read_relation(line))

    nes = [process_named_entities(ne) for ne in nes]

    return sentences, relations, nes, poss

def process_named_entities(nes):
    named_entities = []

    process = False
    for i,ne in enumerate(nes):
        if process and not ne[0] == 'I':
            named_entities[-1]['end'] = i-1
            process = False

        if ne[0] == 'B':
            named_entities.append({'type': ne})
            named_entities[-1]['start'] = i
            process = True

    if process:
        named_entities[-1]['end'] = len(nes)-1

    return named_entities


if __name__ == '__main__':
    sentences, relations, nes, poss = parse_full_re_file('data/kill+birthplace.baseline')
    for i in xrange(len(sentences)):
        print relations[i]