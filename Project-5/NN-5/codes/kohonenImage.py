from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt 

img = plt.imread('./SS/tom.jpg')

print("Training Started")
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))
som = MiniSom(3,3, 3, sigma=0.1, learning_rate=0.3)
starting_weights = som.get_weights().copy()
som.train_random(pixels, 100)
print("Qualization started")
qnt = som.quantization(pixels)

clustered = np.zeros(img.shape)
for i,q in enumerate(qnt):
    clustered[ np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q

print("Done")
plt.figure(1)
plt.subplot(221)
plt.title('Original Image"')
plt.imshow(img)
plt.subplot(222)
plt.title('Vector Quantized Image')
plt.imshow(clustered)
plt.show()





