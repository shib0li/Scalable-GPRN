#!/bin/sh


python main.py --domain jura \
               --kernel ard \
               --device 0 \
               --rank 2 \
               --maxIter 20000 \
               --interval 200 
               

#     args.add_argument("--domain", "-d", dest="domain", type=str, required=True)
#     args.add_argument("--kernel", "-k", dest="kernel", type=str, required=True)
#     args.add_argument("--device", "-v", dest="device", type=str, required=True)
#     args.add_argument("--ranks", "-r", dest="Lrank", type=str, required=True)
#     args.add_argument("--maxIter", "-t", dest="maxIter", type=str, required=True)
#     args.add_argument("--interval", "-i", dest="interval", type=str, required=True)
#     args.add_argument("--initial", '-a', dest="initial", type=str, required=True)
