import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

TRAJ_STEPS = 4096

class Oscillator:

    def __init__(self, eta=0.07674, gamma=1.21, epsilon=12.3):
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

    def dfun(self, state_variables):
        psi1, psi2 = state_variables


        dpsi1 = self.eta * (psi2 - self.gamma * psi1 - psi1**3)
        dpsi2 = self.eta* (- self.epsilon * psi1)

        return [dpsi1, dpsi2]

class ODEintAdapter:
    def __init__(self, model):
        self.model = model

    def dfun(self, state, t):
        return model.dfun(state)


model = Oscillator()
model_ode = ODEintAdapter(model)

y1 = np.linspace(-2.0, 2.0, 20)
y2 = np.linspace(-2.0, 2.0, 20)

Y1, Y2 = np.meshgrid(y1, y2)


u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = model.dfun([x, y])
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     
fig, ax = plt.subplots()

Q = ax.quiver(Y1, Y2, u, v, color='r')

def plot_trajectory(x0, model):
    tspan = np.linspace(0, 200, TRAJ_STEPS)
    ys = odeint(model.dfun, x0, tspan)
    ax.plot(ys[:,0], ys[:,1], 'b-') # path
    ax.plot([ys[0,0]], [ys[0,1]], 'o') # start
    ax.plot([ys[-1,0]], [ys[-1,1]], 's') # end


plot_trajectory([2.,0.], model_ode)

def onclick(event):
    plot_trajectory([event.xdata,event.ydata], model_ode)
    plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
