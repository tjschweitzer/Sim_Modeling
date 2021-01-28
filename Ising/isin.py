import sys
import numpy as np
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class IsingDemon:

    def __init__(self,N,sE):
        self.N = N
        self.demonEnergyDistribution = np.zeros(N*N,dtype=np.int8)
        self.lattice = np.ones((N,N),dtype=np.int8)
        self.systemEnergy = sE  # desired energy value?
        self.demonEnergy = 0
        E= -N*N
        self.magnetization = N*N
        while (E<self.systemEnergy):
            x_rand = random.randint(0,N-1)
            y_rand = random.randint(0,N-1)
            dE = self.deltaE(x_rand,y_rand)
            if (dE>0):
                new_spin = self.lattice[x_rand,y_rand] = -self.lattice[x_rand,y_rand]
                E += dE
                self.magnetization += new_spin*2


        # plt.imshow(self.lattice)
        # plt.title("Delta E {} Initial {} of {}".format(dE,E,self.systemEnergy,))
        # plt.show()

        self.resetData()

    def deltaE (self,x,y):
        # return 2 * self.lattice[x, y]
        return  2*self.lattice[x,y]*(self.lattice[(x-1+self.N)%self.N,y]+\
            self.lattice[(x+1+self.N)%self.N,y]+self.lattice[x,(y-1+self.N)%self.N]+\
            self.lattice[x,(y+1+self.N)%self.N])

    def resetData(self):
        self.mcs = 0
        self.systemEnergyAccumulator = 0
        self.demonEnergyAccumulator = 0
        self.mAccumulator = 0
        self.m2Accumulator = 0
        self.acceptedMoves = 0

    """    public
    void
    doOneMCStep()
    {
        for (int j = 0; j < N;++ j) {
            int i = ( int ) (N * Math.random ( ) );
            int dE = 2 * lattice.getValue ( i, 0) * ( lattice.getValue ( ( i +1) % N, 0) + lattice.getValue ( ( i 􀀀1+N) % N, 0 ) );;
            i f (dE <= demonEnergy ) {
            int newSpin = lattice.getValue ( i, 0 );
            lattice.setValue ( i, 0, newSpin );
            acceptedMoves++;
            systemEnergy += dE;
            demonEnergy = dE;
            magnet izat ion += 2 * newSpin;
            }
            systemEnergyAccumulator += systemEnergy;
            demonEnergyAccumulator += demonEnergy;
            mAccumulator += magnetization;
            m2Accumulator += magnetization * magnet izat ion;
            demonEnergyDistribution[demonEnergy]++;
            }
        mcs++;
    }"""



    def doOneStep(self):
        for j in range(self.N):
            x_rand = random.randint(0,self.N-1)
            y_rand = random.randint(0,self.N-1)
            dE = self.deltaE(x_rand,y_rand)


            if dE <= self.demonEnergy:
                print(self.lattice[x_rand,y_rand],-self.lattice[x_rand,y_rand])
                new_spin = -self.lattice[x_rand,y_rand]
                self.lattice[x_rand, y_rand]= new_spin
                self.acceptedMoves +=1
                # System gives Energy to demon
                self.systemEnergy += dE
                self.demonEnergy -= dE
                self.magnetization += 2*new_spin
                print(x_rand,y_rand, "changed to ", new_spin)
            else:
                print("Nothing Happens")
            # if dE <=0:  # if deltaE is negative
            #     new_spin = -self.lattice[x_rand,y_rand]
            #     self.lattice[x_rand, y_rand]= new_spin
            #     self.acceptedMoves +=1
            #     # System gives Energy to demon
            #     self.systemEnergy += dE
            #     self.demonEnergy -= dE
            #     self.magnetization += 2*new_spin
            #
            # elif (self.demonEnergy>= dE):
            #
            #     # print("{} Taken from to demon".format(dE))
            #     new_spin = self.lattice[x_rand,y_rand] = -self.lattice[x_rand,y_rand]
            #     self.acceptedMoves +=1
            #     # Demon gives energy to system
            #     self.systemEnergy += dE
            #     self.demonEnergy -= dE
            #     self.magnetization += 2*new_spin


            self.systemEnergyAccumulator +=self.systemEnergy
            self.demonEnergyAccumulator +=  self.demonEnergy
            self.mAccumulator += self.magnetization
            self.m2Accumulator += self.magnetization*self.magnetization
            self.demonEnergyDistribution[int(self.demonEnergy)] +=1
            self.mcs += 1
        return self.lattice


    def temperature(self):
        return 4. / np.log(1. + 4. / (self.demonEnergyAccumulator / (self.mcs * self.N)))

app = IsingDemon(5,10)

def main(args):


    size = int(args[0])
    app = IsingDemon(5,10)
    app.doOneStep()
    fig= plt.plot()
    anim = FuncAnimation(fig, app.doOneStep, init_func=init,
                                   frames=200, interval=20, blit=True)

    plt.show()

def init():
    global app


