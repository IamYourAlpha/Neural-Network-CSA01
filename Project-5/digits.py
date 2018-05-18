from sklearn import datasets
from minisom import MiniSom
import matplotlib.pyplot as  plt
# load the digits dataset from scikit-learn
# 901 samples,  about 180 samples per class
# the digits represented 0, 1, 2, 3, 4

digits = datasets.load_digits(n_class=5)
data = digits.data  # matrix where each row is a vector that represent a digit.
num = digits.target  # num[i] is the digit represented by data[i]

som = MiniSom(30, 30, 64, sigma=.8, learning_rate=0.5)
print("Training...")
som.train_random(data, 500)  # random training
print("\n...ready!")
plt.figure(figsize=(7, 7))
wmap = {}
im = 0
for x, t in zip(data, num):  # scatterplot
    w = som.winner(x)
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
                                    color=plt.cm.Dark2(t / 5.))
        

plt.axis([0, som.get_weights().shape[0], 0,
                         som.get_weights().shape[1]])
plt.show()
