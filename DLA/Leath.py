#!/usr/bin/env python



import numpy as np
from IPython import display
import time


class Leath():

    def __init__(self,N,p):
        self.N=N
        self.p =1 - p
        pt = (N//2,N//2)
        
        nn = {(pt[0]+1,pt[1]):True,
              (pt[0]-1,pt[1]):True,
              (pt[0],pt[1]+1):True,
              (pt[0],pt[1]-1):True}
        self.perimeter=nn
        self.cluster = {(N//2,N//2): True}
        self.world = np.zeros((N,N),dtype = np.int8)
        self.perc_flags = [[False,False],[False,False]] #top,bottom,left,right


        
    def add_perimeter(self,pt):
        """
        Give a point pt (touple)
        add to a perimeter all 4 nearest
        neibours that are not already in the cluster
        """
        if pt in self.cluster:
            return

        # if pt[0]<= 0 or pt[0]>self.N:
        #     return False
        # if pt[1]<= 0 or pt[1]>self.N:
        #     return False

        nn = [(pt[0]+1,pt[1]),(pt[0]-1,pt[1]),(pt[0],pt[1]+1),(pt[0],pt[1]-1)]

        for  p in nn:

            if 0 <= p[0] <= self.N and 0 <= p[1] <= self.N :
                if p not in self.cluster and p not in self.perimeter and self.perimeter[pt]:
                    self.perimeter[p] = True

            else:
                print(p,self.perc_flags)
                if p[0]==0: #left side of map
                    self.perc_flags[0][0]=True
                elif p[0]==self.N: #right side of map
                    self.perc_flags[0][1]=True
                if p[1]==0: #top side of map
                    self.perc_flags[1][0]=True
                elif p[1]==self.N: #bottom side of map
                    self.perc_flags[1][1]=True

                if all(self.perc_flags[0])or all(self.perc_flags[0]):
                    print("Perc")
                    return False

    def grow_cluster(self):
      

        new_perim={}
        for p in self.perimeter:

            if self.perimeter[p]:
                new_perim[p]= self.p < np.random.rand()

        for new_pt in new_perim:
            if new_perim[new_pt]:
                if self.add_perimeter(new_pt)  == False:
                    return False
                self.cluster[new_pt]=True
            else:
                self.perimeter[new_pt]=False  #unsure
            
                
                
    def show_board(self):
        

        try:
            for pt in self.cluster:
                    self.world[pt[0],pt[1]]=10

            for pt in self.perimeter:
                if self.world[pt] < 10:
                    if self.perimeter[pt]:
                        self.world[pt[0],pt[1]]=5
                    else:
                        self.world[pt[0],pt[1]]=-1
        except:
            return

   
    def update(self):
        if self.grow_cluster() == False:
            return False

        self.show_board()
        
    
        
    def better_then_FuncAnimation(self):
        plt.figure()
        while True:
            if self.update() == False:
                print("Done update False")
                break
            if all(value == False for value in self.perimeter.values()):
                print("Done All Perim False")
                break


        plt.title("Generation {}".format(i + 1))
        plt.imshow(self.world)
        plt.show()

    def make_a_plot(self):
        point_list = []
        for i in  range(self.N):
            for j in  range(self.N):
                if self.world[i][j]==10:
                    point_list.append([i,j,0])
        distance_from_seed = []
        for point in point_list:
            r = np.sqrt((self.N//2-point[0])**2+(self.N//2-point[1])**2)
            distance_from_seed.append(r)

        unique, counts = np.unique(self.world, return_counts=True)
        my_dict = (dict(zip(unique, counts)))

        unique= np.unique(sorted(distance_from_seed))


        return sorted(distance_from_seed)[1:],unique, my_dict[10]
import matplotlib.pyplot as plt
from scipy.stats import linregress

frames = 100
inital_prob = 0.5927
size = 61
slopes = []
for i in range(2):

    board = Leath(size,inital_prob)

    board.better_then_FuncAnimation()
    distance,my_bins, count = board.make_a_plot()
    x,y = np.histogram(distance,bins=my_bins)

    sum =1
    M=[0]*(len(x))
    for i in range(len(x)):
        sum +=x[i]
        M[i]=sum

# plt.plot(np.log(y[2:]),np.log(M[1:]))
# plt.plot()
# plt.show()


    linreg = linregress(np.log(y[1:]),np.log(M[:]))
    slopes.append( linreg[0])
print(slopes)