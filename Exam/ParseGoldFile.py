__author__ = 'Michael'

import Preprocessing
import Postprocessing

sentences, _, _, poss = Preprocessing.parse_full_re_file('re/train.gold')
Postprocessing.print_sentence_pos_list(sentences, poss)
