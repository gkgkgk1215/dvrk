import matplotlib.pyplot as plt
import numpy as np
import time

class LivePlot():
    def __init__(self, no_sample=1000):
        plt.style.use('ggplot')
        self.fig, self.axes = plt.subplots(nrows=2, figsize=(15, 12))
        self.fig.show()

        self.axes_limit_margin = 0.15;
        self.n = no_sample
        self.cnt = 0
        self.init_flag = False

        # vector initialize
        self.x_vec = []
        self.y1_vec = []
        self.y2_vec = []
        self.y1_axis_max = []
        self.y1_axis_min = []
        self.y2_axis_max = []
        self.y2_axis_min = []

        # graph initialize
        # color = 'red'
        self.line1, = self.axes[0].plot(self.x_vec, self.y1_vec, 'r.-', alpha=0.8)
        self.backgrounds = self.fig.canvas.copy_from_bbox(self.axes[0].bbox)
        self.axes[0].set_xlabel('Sample No.')
        self.axes[0].set_ylabel('Actual Current (jaw 1)')
        self.axes[0].set_xlim([0,self.n])

        # color2 = 'blue'
        self.line2, = self.axes[1].plot(self.x_vec, self.y2_vec, 'r.-', alpha=0.8)
        self.backgrounds2 = self.fig.canvas.copy_from_bbox(self.axes[1].bbox)
        self.axes[1].set_xlabel('Sample No.')
        self.axes[1].set_ylabel('Actual Current (jaw 2)')
        self.axes[1].set_xlim([0,self.n])
        self.fig.canvas.draw()

    def live_plotter(self, y1,y2, title=''):
        if self.init_flag == False:     # run once at the beginning
            self.y1_axis_min = y1
            self.y1_axis_max = y1
            self.y2_axis_min = y2
            self.y2_axis_max = y2
            self.init_flag = True;

        if self.cnt <= self.n*0.9:
            self.x_vec = np.append(self.x_vec, self.cnt)
            self.y1_vec = np.append(self.y1_vec, y1)
            self.y2_vec = np.append(self.y2_vec, y2)
            self.cnt += 1
        else:
            self.y1_vec = np.append(self.y1_vec[1:], y1)
            self.y2_vec = np.append(self.y2_vec[1:], y2)

        if self.y1_axis_min > self.y1_vec.min() or self.y1_axis_max < self.y1_vec.max():
            self.y1_axis_min, self.y1_axis_max = self.set_margin(self.y1_vec, self.axes_limit_margin)
            self.axes[0].set_ylim(self.y1_axis_min, self.y1_axis_max)
            self.fig.canvas.draw()

        self.line1.set_xdata(self.x_vec)
        self.line1.set_ydata(self.y1_vec)
        self.fig.canvas.restore_region(self.backgrounds)
        self.axes[0].draw_artist(self.line1)
        self.fig.canvas.blit(self.axes[0].bbox)

        if self.y2_axis_min > self.y2_vec.min() or self.y2_axis_max < self.y2_vec.max():
            self.y2_axis_min, self.y2_axis_max = self.set_margin(self.y2_vec, self.axes_limit_margin)
            self.axes[1].set_ylim(self.y2_axis_min, self.y2_axis_max)
            self.fig.canvas.draw()

        self.line2.set_xdata(self.x_vec)
        self.line2.set_ydata(self.y2_vec)
        self.fig.canvas.restore_region(self.backgrounds2)
        self.axes[1].draw_artist(self.line2)
        self.fig.canvas.blit(self.axes[1].bbox)

    def set_margin(self, vec, margin):
        vec_max = vec.max() + abs(vec.max() - vec.min()) * margin
        vec_min = vec.min() - abs(vec.max() - vec.min()) * margin
        # if vec.min() >= 0: vec_min = vec.min() * (1 - margin)
        # else: vec_min = vec.min() * (1 + margin)
        # if vec.max() >= 0: vec_max = vec.max() * (1 + margin)
        # else: vec_max = vec.max() * (1 - margin)
        return [vec_min, vec_max]


if __name__ == '__main__':
    p = LivePlot(1000)
    while True:
        y1 = np.random.randn(1) * 1000
        y2 = np.random.randn(1) * 100
        p.live_plotter(y1,y2)
        time.sleep(0.001)