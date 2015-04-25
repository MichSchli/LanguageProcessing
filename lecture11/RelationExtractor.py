__author__ = 'Michael'

import codecs

def read_line(line):
    return line.strip().split('\t')

def read_relation(line):
    return line.strip().split('\t')

def parse_file(filename):
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
        #If the is something:
        else:
            if not parsed_sentence:
                sentence[0].append(read_line(line))
            else:
                sentence[1].append(read_relation(line))

    if not sentence == ([],[]):
        sentences.append(sentence)

    return sentences

def extract_verb_and_entity_combination(sentence):
    verbs = [s for s in sentence if s[2]=='VERB' or s[2]=='AUX']
    named_entities = []

    for s in sentence:
        if s[3][0]=='B':
            named_entities.append([s])
        elif s[3][0]=='I':
            named_entities[-1].append(s)

    return verbs, named_entities

def featurize_verb(verb):
    t = [ord(v) for v in verb[0][1]]
    return sum([t[i]*(30**i) for i in xrange(len(t))])

def featurize_ne(ne):
    return 0

def get_horrible_thingy(verbs, nes):
    l = []
    for v in verbs:
        for n in nes:
            for n2 in nes:
                if n2 != n:
                    l.append(([v],n,n2))
    return l

def p(t):
    print t
    return t

if __name__ == '__main__':
    l = parse_file('kill.data')

    labels = [elem[1] for elem in l]

    thingy = [extract_verb_and_entity_combination(elem[0]) for elem in l]

    other_thingy = [[[[(omgwtf[0], omgwtf[1]) for omgwtf in item] for item in trple] for trple in get_horrible_thingy(*elem)] for elem in thingy]

    #featurized_thingy = [[featurize_verb(tripl[0]) for tripl in sentence] for sentence in other_thingy]

    crappy_prediction = [[('kill' in tripl[0][0][1], tripl[1][0][0], tripl[1][-1][0], tripl[2][0][0], tripl[2][-1][0]) for tripl in sentence] for sentence in other_thingy]

    print crappy_prediction

    #print featurized_thingy[0:3]