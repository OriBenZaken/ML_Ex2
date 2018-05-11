import numpy as np
import pickle

train_x = np.loadtxt("train_x")
train_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")
print "collected"
with open(r"/home/ori/PycharmProjects/ML_Ex2/data_proccessed", "wb") as f:
    f.write(pickle.dumps((train_x, train_y, test_x)))