#!/bin/bash
for f in ~/DataLocal/algo_fin_new/five_min_data/*; do 
	echo "Training on $f ..."; 
	python rlagents_train.py -l False -f data -d $f -t 20000 -u True -m RLG0.pth
done