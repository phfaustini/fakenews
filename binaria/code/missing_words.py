from glob import glob
from fileinput import input

import numpy as np

datasets = {}

current_dataset = None

for line in input('results/setup.txt'):
    line = line.rstrip()
    if line.startswith('Setting up'):
        current_dataset = line.split(' ')[-1]
        datasets[current_dataset] = []
    elif line.startswith('Missing'):
        fraction = line.split(' ')[1] # 28/148
        fraction = fraction.split('/')
        numerator = int(fraction[0])
        divisor = int(fraction[1])
        datasets[current_dataset].append(numerator/divisor)

for k, v in datasets.items():
    print("Dataset {0} missing in average {1:.2f}% of words in each document".format(k, np.mean(v)*100))
