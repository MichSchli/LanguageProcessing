import sys
import codecs
import argparse

def extract_data(file_ner,file_pos,separator=" "):
    """
    extract token-level data (features) from file that is NER tagged and the corresponding POS file 
    this method should output the format expected by the crfsuite script, i.e., nerfeats.py
    that then looks at a tokens context and produces the respective crfsuite format
    
    # Separator of field values.
    separator = ' '

    # Field names ==> this is what this script produces, needs to be aligned with nerfeats.py
    fields = 'w pos y'
    # alternative (add whatever you wanna add and make sure its aligned with nerfeats.py)
    fields = 'w pos cap y'
    """

    # read NER and POS from the two files
    words_tags=read_conll_file(file_ner)
    words_pos=read_conll_file(file_pos)
    
    ## some checks, e.g., that both files have same length, same tokens
    assert(len(words_tags)==len(words_pos))
    
    for (words,tags),(_,pos) in zip(words_tags,words_pos):
        for word,pos,tag in zip(words,pos,tags):
            # first letter is capitalized
            cap="+" if word[0].isupper() else "-"
            hyphen = '+' if '-' in word else '-'
            l = len(word)
            #vowels = "".join(sorted([w for w in word.lower() if w in ['a','e','i','o','u','y']]))
            #################################
            ###### YOUR FEATURES HERE #######  
            #################################
            # 0=separator
            
            ## todo: output the cap feature and more 
            ## make sure the format you output here is what the nerfeats.py script expects as fields!
            print separator.join([word.lower(),pos,cap, l, hyphen, tag])
        # sentence separator
        print ""
        
        
def read_conll_file(file_name):
    """
    read in a file with format:
    word1    tag1
    ...      ...
    wordN    tagN
    
    returns a list of list with (words,tags) tuples for every sentence
    """
    content=[]

    current_words = []
    current_tags = []
    
    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()
        
        if line:
            word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            content.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # if file does not end in newline (it should...), check whether there is an instance in the buffer
    if current_tags != []:
        content.append((current_words, current_tags))
    return content

    
if __name__=="__main__":

    # parse command line options
    parser = argparse.ArgumentParser(description="""Run a structured perceptron""")
    parser.add_argument("--ner", help="NER file (CoNLL format)", required=True)
    parser.add_argument("--pos", help="POS file (CoNLL format)", required=True)
    args = parser.parse_args()

    extract_data(args.ner,args.pos)
