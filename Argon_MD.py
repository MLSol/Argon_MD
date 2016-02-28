# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:03:04 2016

@author: Martijn Sol
"""


from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from math import sqrt, pi

T=1.0    #Temperature
den=0.88 #density
N=864    #Number of particles
dt=4e-3     #size timestep
steps=5000  #Number of timesteps

L=(N/den)**(1/3) #length box
Nhist=200   #Number of bins in histogram
dr=L/(2*Nhist) #bin size

def init_velocity(T,N):
    #Generate velocities from Maxwell distribution in 3D
    v = np.random.normal(0,sqrt(T),(N,3))
    mom = sum(v)
    v -= mom/N
    return v

def init_position(N,L):
    #Make FCC Unit cells
    Nc = round((N/4)**(1/3))
    a=L/Nc
    r=np.zeros((N,3))
    n=0
    for i in range(Nc):
        for j in range(Nc):
            for k in range(Nc):
                #Define particle positions in each unit cell      
                r[n,:]=[i*a,j*a,k*a]
                n += 1
                r[n,:]=[(i+0.5)*a,(j+0.5)*a,k*a]
                n += 1
                r[n,:]=[(i+0.5)*a,j*a,(k+0.5)*a]
                n += 1
                r[n,:]=[i*a,(j+0.5)*a,(k+0.5)*a]
                n += 1                
    return r           

@jit
def force(N,L,r):
    #Calculate the force and potential on each particle
    V=0.0
    F=np.zeros((N,3))
    vir=0.0
    for i in range(r.shape[0]):  #Number of particles
        for j in range(i):
            #Determine in which box the particle is the closest
            dx=r[i,0]-r[j,0]
            dy=r[i,1]-r[j,1]
            dz=r[i,2]-r[j,2]
            dx -= (np.rint(dx/L))*L
            dy -= (np.rint(dy/L))*L
            dz -= (np.rint(dz/L))*L
            
            #Determine vector and distance between particle i and j
            dist2 = dx*dx+dy*dy+dz*dz
            
            dist2 = 1/dist2
            dist6 = dist2*dist2*dist2
            
            #Calculate forces, potential energy and virial
            V += 4*dist6*(-1+dist6)
            F[i,0] -= dist2*dist6*(24-48*dist6)*dx
            F[j,0] += dist2*dist6*(24-48*dist6)*dx
            F[i,1] -= dist2*dist6*(24-48*dist6)*dy
            F[j,1] += dist2*dist6*(24-48*dist6)*dy
            F[i,2] -= dist2*dist6*(24-48*dist6)*dz
            F[j,2] += dist2*dist6*(24-48*dist6)*dz
            vir += dist6*(24-48*dist6)
    
    return F,V,vir

@jit
def update_hist(r,hist,L,dr):
    for i in range(r.shape[0]):  #Number of particles
        for j in range(i):
                #Determine in which box the particle is the closest
                dx=r[i,0]-r[j,0]
                dy=r[i,1]-r[j,1]
                dz=r[i,2]-r[j,2]
                dx -= (np.rint(dx/L))*L
                dy -= (np.rint(dy/L))*L
                dz -= (np.rint(dz/L))*L
                
                #Determine vector and distance between particle i and j
                dist2 = dx*dx+dy*dy+dz*dz
                
                #Update histogram
                if dist2 <= L*L/4:
                    hist[(np.floor(np.sqrt(dist2)/dr))] += 1

    return hist

def simulate(N,L,T,r,v,steps,dt):
    #Simulate the paricles using the velocity-Verlet algorithm
    #Stores the kinectic energy and the virial for each timestep
    Ekin = np.zeros((steps,1))
    virial = np.zeros((steps,1))
    hist=np.zeros((Nhist,1))
    [F,V,vir] = force(N,L,r)
    for i in range(steps):
        v += F*dt/2
        r += v*dt
        r = r%L             
        [F,V,vir] = force(N,L,r)
        v += F*dt/2
        
        Ekin[i] = 0.5*np.sum(v**2)
        virial[i] = vir

        if i%100 == 0:
            hist = update_hist(r,hist,L,dr)
        
    np.savetxt('hist.out' , hist , delimiter=',')
    np.savetxt('Ekin.out' , Ekin , delimiter=',')
    np.savetxt('virial.out' , virial , delimiter=',')
    return r,v
        
def equilibrium(N,L,T,r,v,steps,dt):
    #Simulates the particles while scaling the velocities to achieve
    #a constant temperature
    [F,V,vir] = force(N,L,r)
    for i in range(steps):
        v += F*dt/2
        r += v*dt
        r = r%L             
        [F,V,vir] = force(N,L,r)
        v += F*dt/2
        
        if i%30 == 0:
            la = sqrt((N-1)*3*T/np.sum(v**2))
            v = v*la

    return r,v


#Simulation
r=init_position(N,L)
v=init_velocity(T,N)

[r,v]=equilibrium(N,L,T,r,v,2500,dt)
[r,v]=simulate(N,L,T,r,v,steps,dt)


#Output calculation
hist=np.loadtxt('hist.out', delimiter=',')
Ekin=np.loadtxt('Ekin.out', delimiter=',')
virial=np.loadtxt('virial.out' , delimiter=',')

hist=hist*100/steps
rhist=np.linspace(0,L/2,Nhist)
g = (2*L**3*hist)/(N*(N-1)*4*pi*rhist**2*dr) #correlation function

P=1-np.mean(virial)/(3*N*T) #pressure

Cv=2/(3*N)-np.var(Ekin)/np.mean(Ekin)**2
Cv=1/(N*Cv) #specific heat


#Bootstrap
nt = 1000
Cv_m = np.zeros((nt,1))
P_m = np.zeros((nt,1))
for i in range(nt):
    #A random sample 
    Ekinb = random.sample(list(Ekin),int(steps/2))
    Cv_m[i] = 2/(3*N)-np.var(Ekinb)/np.mean(Ekinb)**2
    Cv_m[i] = 1/(N*Cv_m[i])        
        
    virb = random.sample(list(virial),int(steps/2))
    P_m[i]=1-np.mean(virb)/(3*N*T)
    
Cv_var = np.var(Cv_m)
P_var = np.var(P_m)


#Figures
fig = plt.figure(figsize=(8,6))
plt.plot(rhist, g, linewidth=2.0)
plt.ylabel('g(r)',fontsize=18)
plt.xlabel('r ($\sigma$)',fontsize=18)
plt.tick_params(axis='both',which='major',labelsize=18)
plt.grid()
plt.show()

ti=np.linspace(0,steps*dt,steps)

fig = plt.figure(figsize=(8,6))
plt.plot(ti, 2*Ekin/(3*N))
plt.show()     

#Output
print('T = ',T,'density = ', den)
print('P = ' ,P,'+-',2*sqrt(P_var))
print('Cv = ' ,Cv,'+-',2*sqrt(Cv_var))

