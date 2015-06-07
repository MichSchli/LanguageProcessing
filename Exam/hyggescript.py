import Preprocessing
import itertools
from collections import Counter

sentences, relations, nes, pos = Preprocessing.parse_full_re_file("re/test.gold", zip_ne_to_dictionary=False)


print "sentences " + str(len(sentences))
tokens = list(itertools.chain(*sentences))
final_tokens = len(tokens)
types = len(set(tokens))

more_pos = list(itertools.chain(*pos))
locke_nes = list(itertools.chain(*nes))

more_lations = list(itertools.chain(*relations))
list_lations = [elem['type'] for elem in more_lations]


print "tokens: ", final_tokens
print "types: ", types

print Counter(more_pos).most_common()
print Counter(locke_nes).most_common()
print Counter(list_lations).most_common()


