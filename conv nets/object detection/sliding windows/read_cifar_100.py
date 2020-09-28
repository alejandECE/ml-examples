#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  data_dir = '../../../datasets/classification/cifar-100/'
  # Reads all files paths from data directory
  paths = []
  for root, _, files in os.walk(data_dir):
    for file in files:
      if '.bin' in file:
        paths.append(os.path.join(root, file))

  with open(paths[0], 'rb') as f:
    byte = f.read()
    labels = np.frombuffer(byte, dtype=np.uint8, count=2, offset=0).reshape((2,))
    print(labels)
    img = (np.frombuffer(byte, dtype=np.uint8, count=3072, offset=2)
           .reshape((3, 32, 32))
           .transpose((1, 2, 0))
           )
    plt.imshow(img)
    plt.show()
