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

def parse_full_re_file(filename, zip_ne_to_dictionary=True):
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

    if zip_ne_to_dictionary:
        nes = [process_named_entities(ne) for ne in nes]
    relations = [[process_relations(r) for r in rels] for rels in relations]

    return sentences, relations, nes, poss

def process_relations(rel):

    s1 = rel[0].strip().split(':')
    if len(s1) == 1:
        s1.append(s1[0])

    s2 = rel[1].strip().split(':')
    if len(s2) == 1:
        s2.append(s2[0])

    return {'type': rel[2], 'e1_start': int(s1[0])-1, 'e1_end': int(s1[1])-1, 'e2_start': int(s2[0])-1, 'e2_end': int(s2[1])-1}

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