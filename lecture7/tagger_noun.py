#!/usr/bin/env python
import sys
import codecs

for line in codecs.open(sys.argv[1],encoding="utf-8"):
    line=line.strip()
    if line:
        words=line.split(" ")
        for f in words:
            print "{}\t{}".format(f, "NOUN")
    print ""
