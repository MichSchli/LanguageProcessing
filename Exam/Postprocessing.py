__author__ = 'Michael'



def print_sentence_relation_list(sentences, relation):
    for i, sentence in enumerate(sentences):
        for j,word in enumerate(sentence):
            print str(j+1)+'\t'+word
        print ""
        for r in relation[i]:
            start = str(r['e1_start']+1) + ((':'+str(r['e1_end']+1)) if r['e1_start'] != r['e1_end'] else "")
            end = str(r['e2_start']+1) + ((':'+str(r['e2_end']+1)) if r['e2_start'] != r['e2_end'] else "")

            print start+'\t'+end+'\t'+r['type']
        print ""

def print_sentence_pos_ne_relation_list(sentences, pos, ne, relation):
    for i, sentence in enumerate(sentences):
        for j,word in enumerate(sentence):
            print str(j+1)+'\t'+word+'\t'+pos[i][j]+'\t'+ne[i][j]
        print ""
        for r in relation[i]:
            start = str(r['e1_start']+1) + ((':'+str(r['e1_end']+1)) if r['e1_start'] != r['e1_end'] else "")
            end = str(r['e2_start']+1) + ((':'+str(r['e2_end']+1)) if r['e2_start'] != r['e2_end'] else "")

            print start+'\t'+end+'\t'+r['type']
        print ""

def print_sentence_pos_list(sentences, poss):
    for i, pos in enumerate(poss):
        for j, p in enumerate(pos):
            print sentences[i][j] + '\t' + p
        print ""

def print_sentence_list(sentences):
    for sentence in sentences:
        for word in sentence:
            print word
        print ""