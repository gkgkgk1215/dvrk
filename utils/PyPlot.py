import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    file_name = '../record/jaw_current_peg.txt'
    data_default = np.loadtxt(file_name)
    x = data_default[:,0]
    y = data_default[:,1]

    # plot
    plt.style.use('ggplot')

    plt.figure()
    plt.subplot(211)
    plt.ylabel('Actual current (A)')
    plt.plot(x, 'r')

    plt.subplot(212)
    plt.ylabel('Actual current (A)')
    plt.xlabel('Time (s)')
    plt.plot(y, 'r')
    plt.show()
