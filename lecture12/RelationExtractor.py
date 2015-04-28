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

    return [verbs[0]] if len(verbs) > 0 else [], named_entities

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

def valid_verb(verb):
    return True
    valids = ['kill', 'murder', 'fire', 'assassin', 'shoot']
    for word in valids:
        if word in verb:
            return True
    return False

def entity_type(entity, type): 
   return type in entity[0][-1]

def validate_entities(e1, e2):
   if entity_type(e1, 'PER') and entity_type(e2, 'PER'):
      return 'kill'
   elif (entity_type(e1, 'PER') and entity_type(e2, 'LOC')) or (entity_type(e1, 'LOC') and entity_type(e2, 'PER')):
      return 'birthplace'
   else:
      return ''
   
if __name__ == '__main__':
    l = parse_file('kill+birthplace.data')

    labels = [elem[1] for elem in l]

    thingy = [extract_verb_and_entity_combination(elem[0]) for elem in l]

    other_thingy = [[[[(omgwtf[0], omgwtf[1], omgwtf[3]) for omgwtf in item] for item in trple] for trple in get_horrible_thingy(*elem)] for elem in thingy]

    #print other_thingy
	
    #featurized_thingy = [[featurize_verb(tripl[0]) for tripl in sentence] for sentence in other_thingy]

    crappy_prediction = [[(validate_entities(tripl[1], tripl[2]) if valid_verb(tripl[0][0][1]) else '', tripl[1][0][0], tripl[1][-1][0], tripl[2][0][0], tripl[2][-1][0]) for tripl in sentence] for sentence in other_thingy]

    trues = [[relation for relation in sentence if relation[0]] for sentence in crappy_prediction]
    trues = [[sentence[0]] if len(sentence) > 0 else [] for sentence in trues]
	
    for i in xrange(len(trues)):
        for word in l[i][0]:
            print '\t'.join(word)
        print ''
        for relation in trues[i]:
            firstspan = relation[1] if relation[1] == relation[2] else relation[1]+':'+relation[2]
            secondspan = relation[3] if relation[4] == relation[3] else relation[3]+':'+relation[4] 
            print firstspan+'\t'+secondspan+'\t'+relation[0]
        print ''

    #print featurized_thingy[0:3]