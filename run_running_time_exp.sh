#!/usr/bin/env bash
python=python3
DS=glove-100-angular

for ds in $DS; do
	echo Preparing $ds
	$python cpp/pickle_to_txt.py python/data/$ds.pickle
done

cd cpp/
g++ -O3 -march=native -ffast-math code.cpp -o main

#./main ../python/data/mnist-784-euclidean-data.txt ../python/data/mnist-784-euclidean-queries.txt 784 15 100 1275 4000 | tee mnist-log.txt
#./main ../python/data/sift-128-euclidean-data.txt ../python/data/sift-128-euclidean-queries.txt 128 15 100 270 870 | tee sift-log.txt
./main ../python/data/glove-100-angular-data.txt ../python/data/glove-100-angular-queries.txt 100 15 100 4.7 15.7 | tee glove-log.txt
