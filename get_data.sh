#!/bin/bash
# Downloads the corpus. Train: Shakespeare + Twain. Test: Dickens (held-out author).
set -e
mkdir -p data && cd data
get () { curl -sL -o "$2" "https://www.gutenberg.org/cache/epub/$1/pg$1.txt"; }
curl -sL -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
get 76   twain_huck.txt
get 74   twain_sawyer.txt
get 100  gut_100.txt      # Complete Works of Shakespeare
get 86   gut_86.txt       # A Connecticut Yankee
get 1837 gut_1837.txt     # The Prince and the Pauper
get 3176 gut_3176.txt     # The Innocents Abroad
get 245  gut_245.txt      # Life on the Mississippi
get 3177 gut_3177.txt     # Roughing It
get 119  gut_119.txt      # A Tramp Abroad
get 98   dickens_two_cities.txt
get 1400 dickens_great_expectations.txt
get 1023 dickens_gut_1023.txt   # Bleak House
get 580  dickens_gut_580.txt    # Pickwick Papers
get 730  dickens_gut_730.txt    # Oliver Twist
get 766  dickens_gut_766.txt    # David Copperfield
get 963  dickens_gut_963.txt    # Little Dorrit
echo "done"
