import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Button

TRAJ_STEPS = 4096


class Oscillator:

    def __init__(self, eta=0.07674, gamma=1.21, epsilon=12.3):
        self.limit = (-5, 5), (-5, 5)
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

    def dfun(self, state_variables, *args, **kwargs):
        psi1, psi2 = state_variables


        dpsi1 = self.eta * (psi2 - self.gamma * psi1 - psi1**3)
        dpsi2 = self.eta* (- self.epsilon * psi1)

        return [dpsi1, dpsi2]

class PhasePlaneInteractive:
    def __init__(self, model):
        self.model = model

    def plot_trajectory(self,x0):
        tspan = np.linspace(0, 200, TRAJ_STEPS)
        ys = odeint(self.model.dfun, x0, tspan)
        self.ax.plot(ys[:,0], ys[:,1], 'b-') # path
        self.ax.plot([ys[0,0]], [ys[0,1]], 'o') # start
        self.ax.plot([ys[-1,0]], [ys[-1,1]], 's') # end

    def onclick(self, event):
        self.plot_trajectory([event.xdata,event.ydata])
        plt.draw()

    def __call__(self):
        y1 = np.linspace(-2.0, 2.0, 20)
        y2 = np.linspace(-2.0, 2.0, 20)

        Y1, Y2 = np.meshgrid(y1, y2)


        u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

        NI, NJ = Y1.shape

        for i in range(NI):
            for j in range(NJ):
                x = Y1[i, j]
                y = Y2[i, j]
                yprime = self.model.dfun([x, y])
                u[i,j] = yprime[0]
                v[i,j] = yprime[1]
             
        fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        

        def clear(event):
            self.ax.clear()
            Q = self.ax.quiver(Y1, Y2, u, v, color='r')
            self.ax.set_xlabel('$y_1$')
            self.ax.set_ylabel('$y_2$')
            self.ax.set_xlim(-2.0,2.0)
            self.ax.set_ylim(-2.0,2.0)
            plt.draw()


        clear([0.0, 0.0])
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        plt.xlabel('$y_1$')
        plt.ylabel('$y_2$')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])


        axclear = plt.axes([0.8, 0.02, 0.1, 0.075])
        bclear = Button(axclear, 'Clear')
        bclear.on_clicked(clear)

        plt.show()


if __name__ == '__main__':
    model = Oscillator()
    ppi = PhasePlaneInteractive(model)
    ppi()

