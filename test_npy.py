import numpy as np
print("*****GENERATED DIST*****")
print(np.load('output/dist.npy'))
print("*****EXPECTED DIST*****")
print(np.load('data/processed/dist.npy'))
print("*****GENERATED K*****")
print(np.load("output/K.npy"))
print("*****EXPECTED K*****")
print(np.load('data/processed/K.npy'))

