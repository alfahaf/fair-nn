python=~/miniconda3/bin/python
pypy=~/pypy3.7-v7.3.2-linux64/bin/pypy3

for exp in glove.yml mnist.yaml sift.yaml; do
    $python run.py --exp-file ../exp-files/$exp
done

for exp in lastfm.yml movielens.yml; do
    $pypy run.py --exp-file ../exp-files/$exp
done
