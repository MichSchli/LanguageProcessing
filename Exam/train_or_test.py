import sys
import codecs

def train_or_test(inputfile):
    train_file = open('data/train_data', 'w')
    test_file = open('data/test_data', 'w')
    
    sentences, relations, nes, poss = parse_full_re_file(inputfile)
    # Making a boundary for how much training and testing data we want
    boundary = int(len(sentences)*0.8)

    # Iterating through everything and joining it in the pretty format, we want        
    string = ''
    for i, sentence in enumerate(sentences):
        if i == boundary:
            tempstring = string
            string = ''
        for j,word in enumerate(sentence):
            string += str(j+1)+'\t'+word+'\t'+poss[i][j]+'\t'+nes[i][j]+'\n'
        
        string += '\n'
        
        for r in relations[i]:
            string += '\t'.join(r)+'\n'
        string += '\n'

    # Writing all the stuff into separate test and training files 
    train_file.write(tempstring)
    test_file.write(string)

    train_file.close()
    test_file.close()

def read_relation(line):
    return line.strip().split('\t')

# Parsing the file and returning the sentences, relations, NES'es and POS'es
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

    return sentences, relations, nes, poss

if __name__=="__main__":
    train_or_test('data/kill+birthplace.data')

