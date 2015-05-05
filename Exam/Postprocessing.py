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