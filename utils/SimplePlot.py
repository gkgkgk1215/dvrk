import matplotlib.pyplot as plt
import numpy as np
import CmnUtil as U


if __name__ == '__main__':
    file_name = '../record/jaw_current_needle.txt'
    data_default = np.loadtxt(file_name)

    curr_jaw1 = data_default[:,0]
    curr_jaw2 = data_default[:,1]
    curr_jaw1_filtered = np.zeros(len(curr_jaw1))
    curr_jaw2_filtered = np.zeros(len(curr_jaw2))
    t = np.array(range(len(curr_jaw1)))*0.01  # (sec)

    print t

    fc = 1  # (Hz)
    dt = 0.01 # (ms)
    for i in range(len(curr_jaw1)):
        if i==0:
            curr_jaw1_filtered[i] = curr_jaw1[i]
        else:
            curr_jaw1_filtered[i] = U.LPF(curr_jaw1[i], curr_jaw1_filtered[i-1], fc, dt)

    for i in range(len(curr_jaw2)):
        if i == 0:
            curr_jaw2_filtered[i] = curr_jaw2[i]
        else:
            curr_jaw2_filtered[i] = U.LPF(curr_jaw2[i], curr_jaw2_filtered[i - 1], fc, dt)

    # plot
    plt.style.use('ggplot')

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.title("Repetitive needle grasping", fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Measured current (A)', fontsize=20)
    plt.ylim(-0.15, 0.15)
    plt.plot(t, curr_jaw1_filtered, 'r', label='Jaw1')
    plt.plot(t, curr_jaw2_filtered, 'b', label='Jaw2')
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(loc='upper right', prop={'size': 20})
    plt.show()
