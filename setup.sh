#!/bin/sh

( pip install --upgrade pip                                   || \
  pip3 install --upgrade pip                                  || \
  python -m pip install --upgrade pip                         || \
  python3 -m pip install --upgrade pip )                      && \
( pip install --no-cache-dir -r requirements.txt              || \
  pip3 install --no-cache-dir -r requirements.txt             || \
  python -m pip install --no-cache-dir -r requirements.txt    || \
  python3 -m pip install --no-cache-dir -r requirements.txt )

nbstripout --install --attributes .gitattributes

mkdir data

wget --load-cookies /tmp/cookies.txt -c "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nhHoRAenENHJ3XU_f0rTjHlQHhg4HGx5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nhHoRAenENHJ3XU_f0rTjHlQHhg4HGx5" -O data/irtm.csv && rm -rf /tmp/cookies.txt
