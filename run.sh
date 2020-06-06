#!/bin/sh

python main.py --domain jura \
               --kernel ard \
               --device 0 \
               --rank 2 \
               --maxIter 20000 \
               --interval 200 

