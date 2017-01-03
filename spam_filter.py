# -*- coding: utf-8 -*-
import os
import re
from math import log
from codecs import open
from random import randrange, shuffle
from collections import defaultdict

K = 0.5


def train(samples):
    classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for feats, label in samples:
        classes[label] += 1                 # count classes frequencies
        for feat in feats:
            freq[label, feat] += 1          # count features frequencies

    for label, feat in freq:                # normalize features frequencies
        freq[label, feat] /= classes[label]
    for c in classes:                       # normalize classes frequencies
        classes[c] /= len(samples)

    return classes, freq                    # return P(C) and P(O|C)


def classify(classifier, feats):
    classes, prob = classifier
    return min(
        classes.keys(),
        key=lambda cl: -log(classes[cl]) + sum(
            -log(prob.get((cl, feat), 10**(-7))) for feat in feats
        )
    )


def get_features(sample):
    return re.split(r'\W+', sample)


def main():
    hams = (
        open(entry.path, 'r', 'utf-8', 'ignore').read()
        for entry in os.scandir(r'data/ham')
        if entry.is_file()
    )
    features = [(get_features(feat), 'ham') for feat in hams]
    spams = (
        open(entry.path, 'r', 'utf-8', 'ignore').read()
        for entry in os.scandir(r'data/spam')
        if entry.is_file()
    )
    features += [(get_features(feat), 'spam') for feat in spams]
    shuffle(features)
    samples = []
    for _ in range(round(len(features) * K)):
        samples.append(features.pop(randrange(len(features))))

    classifier = train(features)

    mistakes = sum(map(
        lambda sample: 0 if classify(classifier, sample[0]) == sample[1] else 1,
        samples
    ))
    print(mistakes, mistakes/len(samples))


if __name__ == '__main__':
    main()
