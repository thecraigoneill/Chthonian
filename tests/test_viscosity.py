import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

m=100

T_top=0.01
T_base=1.01 # 0.001
T_ave = (T_top + T_base)/2.0

z1=np.linspace(660e3,0,m+1)
# Age of litho being thinned - 60 Myr here.
T1 = 0.9*erf((z1)/(2*np.sqrt(1e-6*(60*1e6*60*60*24*365.25)))) + T_top




#Scaling 
ra=5e6 #4.167e4 # Might be slightly subcritical if ~1e4
pr=1e3 #Mora 23
grav=0.1 # Using the scaling of Mora et al. 2023
beta=0.001

gbeta = grav*beta
visco2 = (gbeta*(T_base - T_top) * (m-1)**3 )/(ra * 1/pr) 
visco = np.sqrt(visco2)

print("Viscosity!",visco)


R=1
E=5.5
v0=1
fn = np.exp(E/(R*T1 + 1.e-3))
fn2= np.max(fn)
fn = fn/fn2

#print("fn",fn)

visc = visco * fn

# Mora formulation
b = 5*E/T_ave
print("b",b)
visc2 = np.exp(b * (T_ave/T1 - 1)/(T_ave/T_top - 1))
v0 = (visco/np.min(visc2))
visc2 = visc2 * v0
visc = visc2
visc[visc>1e4] = 1.0e4

plt.subplot(121)
plt.plot(T1,z1/1e3,color="xkcd:blood red")
plt.ylim(np.max(z1/1e3),0)


plt.subplot(122)
plt.plot(np.log10(visc),z1/1e3,color="xkcd:blood red")
plt.ylim(np.max(z1/1e3),0)
#plt.xlim()
print(np.c_[z1/1e3,T1,visc])

plt.show()

