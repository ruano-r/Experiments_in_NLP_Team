#simple script to turn the hungarian data (separating morphemes with "-") into separations with "@@"

import sys

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        outfile.write(line.replace("-", "@@"))
        