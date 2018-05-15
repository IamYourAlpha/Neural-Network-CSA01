from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    x/np.linalg.norm(x)

data = np.genfromtxt('iris.csv', delimiter=',', usecols=(0,1,2,3))
#data = np.apply_along_axis(fun, 1, data)
som = MiniSom(10, 10,4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
print som._weights

som.train_random(data, 1000)

plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()

target = np.genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
t = np.zeros(len(target), dtype=int)
t[ target == 'Iris-setosa' ] = 0
t[ target == 'Iris-versicolor' ] = 1
t[ target == 'Iris-virginica' ] = 2

markers = [ 'o', 's', 'D']
colors = ['r', 'g', 'b']
inc = 0.5
for cnt, d in enumerate(data):
    w = som.winner(d)
    plt.plot(w[0]+inc, w[1]+inc, markers[ t[cnt] ], markerfacecolor='None',
    markeredgecolor=colors[ t[cnt] ], markersize=12, markeredgewidth=2 )

print som.quantization_error
plt.show()

