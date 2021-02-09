import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
from  pymbar.timeseries import detectEquilibration

class IsingDemon:

    def __init__(self, L):
        self.L = L
        self.N = L * L
        self.lattice = np.ones((L, L), dtype=np.int8)
        self.critTemp = 2. / np.log(1. + np.sqrt(2.))
        self.temperature = self.critTemp
        self.mcs = 0
        self.energy = 0
        self.energyAccumulator = 0
        self.energySquaredAccumulator = 0
        self.magnetization = 0
        self.magnetizationAccumulator = 0
        self.magnetizationSquaredAccumulator = 0
        self.acceptedMoves = 0
        self.w = np.zeros(9)

        self.magnetization = self.N
        self.energy = -2 * self.N
        self.resetData()

        self.w[8] = np.exp(-8. / self.critTemp)  # is doing a tayor series expensive
        self.w[4] = np.exp(-4. / self.critTemp)

    def resetData(self):
        self.mcs = 0
        self.systemEnergyAccumulator = 0
        self.demonEnergyAccumulator = 0
        self.mAccumulator = 0
        self.m2Accumulator = 0
        self.acceptedMoves = 0
        self.magnetization_data = []
        self.systemEnergy_data = []
        self.my_temp = []
        self.susceptibility_data = []
        self.move_prob = []

    def randomLattice(self):
        self.lattice = np.where( np.random.random((self.L, self.L))<.5,-1,1)
        self.magnetization=np.sum(self.lattice)
        self.energy= -2. *self.magnetization


    def specificHeat(self):
        energySquaredAverage = self.energySquaredAccumulator / self.mcs
        energyAverage = self.energyAccumulator / self.mcs
        heatCapacity = energySquaredAverage - energyAverage * energyAverage
        heatCapacity = heatCapacity / (self.temperature * self.temperature)

        return heatCapacity / self.N

    def changeTemperature(self, newTemperature):
        self.temperature = newTemperature
        self.w[8] = np.exp(-8. / self.temperature)
        self.w[4] = np.exp(-4. / self.temperature)

    def susceptibility(self):
        magnetizationSquaredAverage = self.magnetizationSquaredAccumulator / self.mcs
        magnetizationAverage = self.magnetizationAccumulator / self.mcs
        return (magnetizationSquaredAverage - magnetizationAverage * magnetizationAverage) / (self.temperature * self.N)

    def deltaE(self, i, j):
        return 2 * self.lattice[i, j] * (self.lattice[(i + 1) % self.L, j] + self.lattice[(i - 1 + self.L) % self.L, j]
                                         + self.lattice[i, (j + 1) % self.L] + self.lattice[
                                             i, (j - 1 + self.L) % self.L])

    def doOneMCStep(self):
        for k in range(self.N):
            i = np.random.randint(self.L - 1)
            j = np.random.randint(self.L - 1)

            dE = self.deltaE(i, j)
            if (0 >= dE) or self.w[dE] > np.random.random():
                newSpin = self.lattice[i, j] = -self.lattice[i, j]
                self.acceptedMoves += 1
                self.energy += dE
                self.magnetization += 2 * newSpin

            # only do stats every N steps?
            self.energyAccumulator += self.energy
            self.energySquaredAccumulator += self.energy * self.energy
            self.magnetizationAccumulator += self.magnetization
            self.magnetizationSquaredAccumulator += self.magnetization * self.magnetization

        self.mcs += 1
        self.my_temp.append(self.specificHeat())
        self.magnetization_data.append(self.magnetizationAccumulator/(self.mcs * self.N))
        self.move_prob.append(self.acceptedMoves / (self.mcs * self.N))
        self.susceptibility_data.append(self.susceptibility())

        self.systemEnergy_data.append(self.energyAccumulator/(self.mcs * self.N))

def check_equilibration(heat,mag,suscept,move_prob, flux=.02):


    print("S_Heat {}".format(detectEquilibration(np.array(heat))))

### Animation code thanks to Anthony!
def animate(counter):
        #used to save graphs as pngs
    if counter % 50 == 49:
        check_equilibration(ising_model.my_temp,mag_line,susceptibility_line,move_prob_line)
        # plt.savefig("Data{}.png".format(counter))
        # plt.title("Random Spins")
        # ising_model.resetData()
        # ising_model.randomLattice()


    # Run a step in the simulation and display the simulation image
    ising_model.doOneMCStep()
    im.set_data(ising_model.lattice)
    # Plot temp, magnetism, and system energy over monte carlo steps
    x_data = np.arange(1, ising_model.mcs + 1)
    heat_line.set_data(x_data, ising_model.my_temp)
    mag_line.set_data(x_data, ising_model.magnetization_data)
    susceptibility_line.set_data(x_data, ising_model.susceptibility_data)
    energy_temp_line.set_data(x_data,ising_model.systemEnergy_data)
    move_prob_line.set_data(x_data,ising_model.move_prob)
    # Reset the scale and limits of the plots
    ax[0, 0].relim()
    ax[0, 0].autoscale_view(True, True, True)
    ax[0, 1].relim()
    ax[0, 1].autoscale_view(True, True, True)
    ax[0, 2].relim()
    ax[0, 2].autoscale_view(True, True, True)

    ax[1, 0].relim()
    ax[1, 0].autoscale_view(True, True, True)
    ax[1, 1].relim()
    ax[1, 1].autoscale_view(True, True, True)
    ax[1, 2].relim()
    ax[1, 2].autoscale_view(True, True, True)






N = 20

ising_model = IsingDemon(N)
# Initialize the dashboard
fig, ax = plt.subplots(2, 3)
im = ax[0, 0].imshow(ising_model.lattice, cmap='Greys', vmin=-1,vmax=1)
heat_line, = ax[0, 1].plot([], [], lw=3)

fig.suptitle("All Upspins", fontsize=16)
move_prob_line, = ax[0, 2].plot([], [], lw=3)
mag_line, = ax[1, 0].plot([], [], lw=3)
susceptibility_line, = ax[1, 1].plot([], [], lw=3)

energy_temp_line, = ax[1, 2].plot([], [],'r', lw=3)
ax[0, 0].set_title('System')
ax[0, 1].set_title('Specific Heat')
ax[0, 2].set_title('Move Probability')

ax[1, 0].set_title('Magnetization')
ax[1, 1].set_title('Susceptibility')
ax[1, 2].set_title('Energy')
# ax[1, 0].set_title('Magnetization')

ising_model.randomLattice()
line_ani = animation.FuncAnimation(fig, animate,1000,repeat=False )

plt.show()


ising_model = IsingDemon(N)
# Initialize the dashboard
fig, ax = plt.subplots(2, 3)
im = ax[0, 0].imshow(ising_model.lattice, cmap='Greys', vmin=-1,vmax=1)
heat_line, = ax[0, 1].plot([], [], lw=3)

fig.suptitle("All Upspins Temp set to 2.", fontsize=16)
move_prob_line, = ax[0, 2].plot([], [], lw=3)
mag_line, = ax[1, 0].plot([], [], lw=3)
susceptibility_line, = ax[1, 1].plot([], [], lw=3)

energy_temp_line, = ax[1, 2].plot([], [],'r', lw=3)
ax[0, 0].set_title('System')
ax[0, 1].set_title('Specific Heat')
ax[0, 2].set_title('Move Probability')

ax[1, 0].set_title('Magnetization')
ax[1, 1].set_title('Susceptibility')
ax[1, 2].set_title('Energy')
# ax[1, 0].set_title('Magnetization')

ising_model.changeTemperature(2.)
line_ani = animation.FuncAnimation(fig, animate,1000,repeat=False )

plt.show()


