import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    file_name = "jaw_current_peg.txt"
    assert os.path.exists(file_name), \
        'The file does not exist: {}'.format(file_name)
    data = np.loadtxt(file_name, delimiter=' ')

    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=2, figsize=(15, 12))
    plt.show()

    plt.plot(data[:, 0])

