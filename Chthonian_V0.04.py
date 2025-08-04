import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from numba import jit
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit, prange
from scipy.special import erf
import shutil
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# UPDATES
# Includes T-dep visc, and plasticity
# Check volc against Un version


###################### MIT LICENSING ###########################################
#
#
#Copyright 2024 Dr Craig O'Neill
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################ 

########################################################################################################################################################
# The following is a minimum-requisite and dependency python script to calculate mantle flow field under Hadean conditions after an impact strike.     #
# It calculates volcanic heat transport, and transport thermal effects. The thermal convection solver has been benchmarked. Full implementation is to  #
# presented in expanded upcoming publications. Please see https://github.com/thecraigoneill/planet_LB for the core code archive.                       #
########################################################################################################################################################



# Use 1D melt column routine as basis
# Loop through each column in 2D to calculate
# Initialisation?

# Melt routine - 1D
# Import columns of model seperately.
@jit(nopython=True)
def magma_source_1Dcol(T,C,Y,x_Ts,x_Tl,dx,dt,dens,melt_dens):
    Volume = dx # 1D
    # Get existing depletion as depleted solidus
    Cmod = -1 * np.maximum(C,np.ones_like(C)*(-1))  #Depletion should be tracked as -1 < C < 0. Positive for crust.
    Cmod[Cmod < 0] = 0.0  # Eliminate crust from depleting
    Depl_sol = Cmod * (x_Tl - x_Ts) + x_Ts 
    Depl_melt_frac = (T - Depl_sol)/(x_Tl-x_Ts)
    Depl_melt_frac[Cmod < 0] = 0.0 # Eliminate crustal melting... again.
    #if(np.any(np.isnan(C))):
    #   print("NAN in C")
    #   #exit()

    ###### Melt Fraction ###########################
    #melt_frac = ((T - x_Ts)/(x_Tl - x_Ts)) #As fraction
    #print(melt_frac[melt_frac > 0])
    #melt_frac [melt_frac  < 0] = 0.0
    #melt_frac [melt_frac  > 1] = 1.0
    # Now check against depl, and modify f accordingly
    melt_frac2 = np.copy(Depl_melt_frac) #melt_frac - Depl_old
    melt_frac2[melt_frac2  < 0] = 0.0
    melt_frac2[melt_frac2  > 1] = 1.0
    melt_frac = np.copy(melt_frac2)

    #if(np.any(np.isnan(melt_frac))):
    #   print("NAN in Melt frac")


    #VOLC
    Volc1 = np.copy(melt_frac2)*0.0  # Purely for source region here, not injection yet.
    Depl = np.copy(melt_frac2)
    Depl[C > 0.6] = 0.0
    
    Tmagma = np.zeros_like(Y)
    #dens2 = np.zeros_like(dens)
    devs = np.random.normal(0,50,np.shape(dens))  #This is dimensional
    dens2 = np.copy(dens) + devs
    done=0
    gamma = np.random.gamma(1,1,np.shape(Volc1))
    melt_den = melt_dens + (-150.0*gamma) + 80  #This is dimensionalised
    melt_den[melt_frac == 0] = 0.0

    Volc2 = np.copy(Volc1[:])  #Note arrays get tranposed when plotted.
    dens2a = np.copy(dens2)
    T1 = np.copy(T[:])
    Tmagma1 = np.copy(Tmagma[:])
    melt_den1 = np.copy(melt_den[:])



    for i in range(len(Volc1)):
        # Need to propagate up from melt depth/density to surface
        # Note in 2D Y[:,0]: 0->1
        #            dens: 2900 -> 3400
        #            melt_dens -> surface-> base
        if( (melt_frac[i] > 0.0)&( (C[i]>=-1)&(C[i]<0.6)) ):   #Limit melt transport to fertile mantle
            dens_col = (dens2a[i:]) #[::-1] # density array from melt depth upwards 
            y = Y[i:]
            done=0
            kk=0 #
            #print("Melt:",melt_frac[i],Y,dens_col,melt_den1[i])
            for j in range(len(dens_col)): #Includes last index. Bottom up search. 
              #print("Melting:",i,j,kk,Y[i],y[j],melt_frac[i],melt_den1[i],dens_col[j])
              if ((done == 0) & (melt_den1[i] > dens_col[j])): #Emplaced at neutral buoyancy
                done=1
                kk=j
                #print(kk)
              elif((done==0)&(j==(len(dens_col))-1 )):  #Erupted at surface
                kk=j-1
                done=1
                #print("2:",kk)

            if (kk>0):
                ii = i + kk # Reconstruct initial index
                Volc2[ii] += (melt_frac[i]) #THis should invert Volc again to +ve
                Tmagma1[ii] = T1[i]
                #print(i,kk,ii,"Depths melt/emplacemnt:",Y[i],Y[ii],"Melt/volc:",melt_frac[i],Volc2[ii],"C",C[i],"Depl",Depl[i],"Dens melt/crust:",melt_den1[i],dens2a[ii])

    Volc1 = np.copy(Volc2)
    #print("Melt",melt_frac,"Volc:",Volc1,"Depl:",Depl)
    #print("VOLC:",Volc)
    dens2 = dens2a      
    #T[k,:] = T1      
    Tmagma = np.copy(Tmagma1) 
    melt_den = melt_den1
    #print("VOLC",Volc[0:10,0:10],"Tmagma",Tmagma[0:10,0:10],"is nan?",np.isnan(Volc),np.isnan(Tmagma))

    if(np.any(np.isnan(Volc1))):
       #print("NAN in Volc")
       Volc1[np.isnan(Volc1)] = 0.0
    if(np.any(np.isnan(Tmagma))):
       #print("NAN in Tmagma")
       Tmagma[np.isnan(Tmagma)] = 0.0
    Tmagma[-1] = 0.0
    #Volc[-1] = 0

    #print("AMI here",np.shape(Volc1),np.shape(Depl))
    return melt_frac, Volc1, Depl, Tmagma

@jit(nopython=True)
def loop_over_melt_columns(th,C,Y,x_Ts,x_Tl,dx,dt,dens,melt_dens):
    melt_frac = np.zeros_like(Y)
    melt_den = np.ones_like(Y)*melt_dens
    Depl = np.zeros_like(Y)
    Volc = np.zeros_like(Y)
    Tmagma = np.zeros_like(Y)
    for k in range(len(Volc[:,0])):
        dens2 = dens[k,:]
        C2 = C[k,:]
        Y2 = Y[k,:]
        #T12 = np.copy(th[k,:])
        Ts2 = np.copy(x_Ts[k,:])
        T2 = np.copy(th[k,:])
        Tl2 = np.copy(x_Tl[k,:])
        melt_den2 = melt_den[k,:]
        #print("SHAPE1",np.shape(Volc[k,:]),np.shape(melt_den2))
        melt_frac[k,:], Volc[k,:], Depl[k,:], Tmagma[k,:] =  magma_source_1Dcol(T2,C2,Y2,Ts2,Tl2,dx,dt,dens2,melt_den2)
    return  melt_frac, Volc, Depl, Tmagma




@jit(nopython=True)
def loopy_loop(n,m,vx,vy,cx,cy,rho,w,feq):
    for i in range(0,n+1):
        for j in range(0,m+1):
            t1 = vx[i,j]*vx[i,j] + vy[i,j]*vy[i,j]
            for k in range(0,9):
                t2 = vx[i,j]*cx[k] + vy[i,j]*cy[k]
                feq[k,i,j] = rho[i,j]*w[k]* (1.0+3.0*t2+4.50*t2*t2-1.50*t1) # Chekc for diffusion too.
    return feq

# @jit(nopython=True)
def LB_D2Q9_V(dt,m,n,gbeta,cx,cy,omegaV,w,rho,th,vx,vy,f,feq,u0, Volc, Tmagma):
    time=0
    #t1 = vx*vx + vy*vy
    #t2 =  vx[:,:]*cx[0] + vx[:,:]*cx[1] + vx[:,:]*cx[2] + vx[:,:]*cx[3] + vx[:,:]*cx[4] +vx[:,:]*cx[5] + vx[:,:]*cx[6] +vx[:,:]*cx[7]+vx[:,:]*cx[8]
    #t2 += vy[:,:]*cy[0] + vy[:,:]*cy[1] + vy[:,:]*cy[2] + vy[:,:]*cy[3] + vy[:,:]*cy[4] +vy[:,:]*cy[5] + vy[:,:]*cy[6] +vy[:,:]*cy[7]+vy[:,:]*cy[8]
    # This next bit is a bit bespoke.........
    #tref = 0.5
    tref = (np.min(th) + np.max(th))/2

    #tmp = th - tref
    #l=50
    #/4000.0 # Rescaled to non-dim between 0 - 4000 kg/m3
    #print("Chekcing dT:",np.min(tmp),np.max(tmp),np.mean(tmp),gbeta,w[2],cy[2])
    # Gravity force term
    F= np.zeros_like(feq)
    cdens = 0.5 + (2900 - dens)/(3400 - 2900)  # Buoyancy terms need to be regularised against range.
    gbeta10 = gbeta*0 # This is determined by force balance. Can cause initial conditions to bounce!
    F[0,:,:] = 3. * w[0] * gbeta * (th - tref) * cy[0] * rho + 3. * w[0] * gbeta10 * cdens * cy[0] * rho
    F[1,:,:] = 3. * w[1] * gbeta * (th - tref) * cy[1] * rho + 3. * w[1] * gbeta10 * cdens * cy[1] * rho
    F[2,:,:] = 3. * w[2] * gbeta * (th - tref) * cy[2] * rho + 3. * w[2] * gbeta10 * cdens * cy[2] * rho
    F[3,:,:] = 3. * w[3] * gbeta * (th - tref) * cy[3] * rho + 3. * w[3] * gbeta10 * cdens * cy[3] * rho
    F[4,:,:] = 3. * w[4] * gbeta * (th - tref) * cy[4] * rho + 3. * w[4] * gbeta10 * cdens * cy[4] * rho
    F[5,:,:] = 3. * w[5] * gbeta * (th - tref) * cy[5] * rho + 3. * w[5] * gbeta10 * cdens * cy[5] * rho
    F[6,:,:] = 3. * w[6] * gbeta * (th - tref) * cy[6] * rho + 3. * w[6] * gbeta10 * cdens * cy[6] * rho
    F[7,:,:] = 3. * w[7] * gbeta * (th - tref) * cy[7] * rho + 3. * w[7] * gbeta10 * cdens * cy[7] * rho
    F[8,:,:] = 3. * w[8] * gbeta * (th - tref) * cy[8] * rho + 3. * w[8] * gbeta10 * cdens * cy[8] * rho
    #print("Checking forces:",F[2,l,m-l],  "T force:", 3.*w[2]*gbeta*(th[l,m-l]-tref)*cy[2]*rho[l,m-l],rho[l,m-l],th[l,m-l], "C force:", 3.*w[2]*gbeta10*cdens[l,m-l]*cy[2]* rho[l,m-l],cdens[l,m-l],dens[l,m-l])
    #print("Checking force",F[2,:,m-l])
    #Collision step
    feq = loopy_loop(n,m,vx,vy,cx,cy,rho,w,feq)

    # Assumes nonDim Tmagma (see magma_source_sol routine), and temperature th. 
    #dT = Tmagma - th # Heat added.
    #dT[dT < 0] = 0
    #dT2 = (dT/dt) * Volc #This term can get quite large

    #dT = np.minimum((Tmagma - th),(x_Ts - th)) #Non-dim Tmagma. Prevents runaway melt silliness.
    #dT2 = np.zeros_like(dT)
    #dT2=dT/dt
    #B=np.abs(Volc)
    filt4 = Volc>1.0
    Volc[filt4] = 1.0
    filt4 = Volc<-1.0
    Volc[filt4] = -1.0
    eff_fac = 5e-4
    VSource = eff_fac*Volc/dt # Erupted volcanic/magmatic source term. We need to moderate the change for each timestep.
    #dT2[~vfilt1] = 1e-2 * dT[~vfilt1]/dt 

    f = (1.0 - omegaV)*f + omegaV*feq + F + dt*0.5*VSource   #
    
    #Streaming step  # 

    f[1,:,:] = np.roll(f[1,:,:],1,axis=0)
    f[2,:,:] = np.roll(f[2,:,:],1,axis=1)
    f[3,:,:] = np.roll(f[3,:,:],-1,axis=0)
    f[4,:,:] = np.roll(f[4,:,:],-1,axis=1)

    f[5,:,:] = np.roll(f[5,:,:],1,axis=0)
    f[5,:,:] = np.roll(f[5,:,:],1,axis=1)
    f[6,:,:] = np.roll(f[6,:,:],-1,axis=0)
    f[6,:,:] = np.roll(f[6,:,:],1,axis=1)
    f[7,:,:] = np.roll(f[7,:,:],-1,axis=0)
    f[7,:,:] = np.roll(f[7,:,:],-1,axis=1)    
    f[8,:,:] = np.roll(f[8,:,:],1,axis=0)
    f[8,:,:] = np.roll(f[8,:,:],-1,axis=1)
  
    #print("Checking f2",f[2,:,m-l])


    # Bounce back
    # Left/West
    f[1,0,:]=f[3,0,:]
    f[5,0,:]=f[7,0,:]
    f[8,0,:]=f[6,0,:]
    # Right/East
    f[3,n,:]=f[1,n,:]
    f[7,n,:]=f[5,n,:]
    f[6,n,:]=f[8,n,:]
    # Bottom
    f[2,:,0]=f[4,:,0]
    f[5,:,0]=f[7,:,0]
    f[6,:,0]=f[8,:,0]
    # Top - Free slip; Change to 478->256 for bounceback.
    f[4,:,m]=f[2,:,m]
    f[7,:,m]=f[6,:,m]
    f[8,:,m]=f[5,:,m]

    #print("Checking f3",f[2,:,m-l])


    rho = f[0,:,:] + f[1,:,:] + f[2,:,:] + f[3,:,:] + f[4,:,:] +f[5,:,:] + f[6,:,:]+f[7,:,:]+f[8,:,:]
    
    # Fixed velocity boundary condition
    # Note that u0 can be a single float, or an array of length n
    # If you is not numeric but a string, we assume free slip
    #if (type(u0) == np.ndarray)or(type(u0) == float):
    #    rhon = (f[0,:,m] + f[1,:,m] + f[3,:,m] + 2.0*(f[2,:,m] + f[6,:,m] + f[5,:,m]))  # C/W page 213 of Mohamad, he has 9, 1 ,3 instead of 0, 1, 3 as previous. Schema?
    #   f[8,:,m] = f[6,:,m] + rhon*u0/6.0
    #    f[7,:,m] = f[5,:,m] - rhon*u0/6.0  # Boundary condition affect here
    #    rho[:,m] = (f[0,:,m] + f[1,:,m] + f[3,:,m] + 2.0*(f[2,:,m] + f[6,:,m] + f[5,:,m]))  # C/W page 213 of Mohamad, he has 9, 1 ,3 instead of 0, 1, 3 as previous. Matlab v python?

    #prho= f[0,:,:] + f[1,:,:] + f[2,:,:] + f[3,:,:] + f[4,:,:] +f[5,:,:] + f[6,:,:]+f[7,:,:]+f[8,:,:]


    #np.set_printoptions(precision=3)
    #print("\tf\t", f[0,0:10,10])
    #print("\trho\t",  rho[0:10,10])

    usum = f[0,:,:]*cx[0] + f[1,:,:]*cx[1] + f[2,:,:]*cx[2] + f[3,:,:]*cx[3] + f[4,:,:]*cx[4] +f[5,:,:]*cx[5] + f[6,:,:]*cx[6] +f[7,:,:]*cx[7]+f[8,:,:]*cx[8]
    vsum = f[0,:,:]*cy[0] + f[1,:,:]*cy[1] + f[2,:,:]*cy[2] + f[3,:,:]*cy[3] + f[4,:,:]*cy[4] +f[5,:,:]*cy[5] + f[6,:,:]*cy[6] +f[7,:,:]*cy[7]+f[8,:,:]*cy[8]
    vx = usum/rho
    vy = vsum/rho
    #print("Checking vy",vy[:,m-l])

    #print("\tvx\t",  vx[0:5,10])
    #rho *= 4000.0 #rescale after the fact for melting calcs
    if(np.any(np.isnan(vx))):
       print("NAN in vx")
    if(np.any(np.isnan(vy))):
       print("NAN in vy")
    if(np.any(np.isnan(rho))):
       print("NAN in rho")

    fneq = f - feq
    return rho, vx, vy, f, fneq

# @jit(nopython=True)
def LB_D2Q9_T(dt,m,n,cx,cy,omegaT,w,T_top,T_base,T,vx,vy,g,geq,H, Volc, Tmagma, x_Ts, Depl, Cp, Y):
    #Collision
    np.set_printoptions(precision=3)
    th=T
    #print("\t\tvx\t",  vx[0:5,10])
    #print("\t\tth\t",  th[0:10,10])
    #print("\t\tw,cx,cy,omegaT:\t",w[0],cx[0],cy[0],omegaT)
    geq[0,:,:] = w[0]*th*(1.0+3.0*(vx*cx[0]+ vy*cy[0]))
    geq[1,:,:] = w[1]*th*(1.0+3.0*(vx*cx[1]+ vy*cy[1]))
    geq[2,:,:] = w[2]*th*(1.0+3.0*(vx*cx[2]+ vy*cy[2]))
    geq[3,:,:] = w[3]*th*(1.0+3.0*(vx*cx[3]+ vy*cy[3]))
    geq[4,:,:] = w[4]*th*(1.0+3.0*(vx*cx[4]+ vy*cy[4]))
    geq[5,:,:] = w[5]*th*(1.0+3.0*(vx*cx[5]+ vy*cy[5]))
    geq[6,:,:] = w[6]*th*(1.0+3.0*(vx*cx[6]+ vy*cy[6]))
    geq[7,:,:] = w[7]*th*(1.0+3.0*(vx*cx[7]+ vy*cy[7]))
    geq[8,:,:] = w[8]*th*(1.0+3.0*(vx*cx[8]+ vy*cy[8]))  # rho * energy_dens for th. - PM24
   


    # Depletion effect - refridgeration
    prefac_D = 1e-4
    dT2 = np.zeros_like(T)
    dT1 = np.copy(dT2)

    dT_D =  (T - x_Ts)  # Temperature above solidus
    vfilt1 = (-1*Depl < C) # Has depletion exceeded previous depletion (as tracked by C)? 
    f3 = dT_D > 0
    dT_D[~f3] = 0.0 #If not above solidus temperature, no depletion (this is a sanity check)
    dT_D[~vfilt1] = 0.0 # If not exceed previous depletion ie. C, no depletion.

    mass_D = np.zeros_like(dT_D)
    mass_D[vfilt1] = np.abs(Depl[vfilt1] - C[vfilt1]) * dens[vfilt1] # Mass of newly depleted material
    J2 = Cp * mass_D * dT_D  # Energy consumed depleting that mass
    fmD = mass_D > 0  #Only consider material depleted
    dT1[fmD] = -1*np.abs( (J2[fmD]/(mass_D[fmD] * Cp + 1e-9) ) ) * 1/dt      # Energy change total
    dT1 *= prefac_D  # Limit by efficiency of melt transfer (assumed 1 here)


    # Now add volc heat terms
    prefac_V = 1e-4
    filT = Tmagma > 0
    dT = Tmagma - T
    dT[dT < 0] = 0.0 #np.maximum((Tmagma[filT] - th[filT]),(x_Ts[filT] - th[filT]))
    filtT2 = T > x_Tl
    dT[filtT2] = 0 # Can't add more heat if at liquidus (here at least)
    mass = Volc * dens # mass of intruded material in kg* B) # Accounts for temperature and volume fraction of melt
    J1 = Cp * mass * dT #Energy added
    fm = mass > 0
    dT2[fm] = np.abs( ( J1[fm]/(mass[fm] * Cp) )) * 1/dt
    dT2 *= prefac_V
    fm2 = (mass == 0)
    dT2[fm2] = 0.0

    # Catch runaway thermals
    sink = T > (x_Tl)
    prefac= 0.0*1e-15
    Sink = np.zeros_like(T)
    Sink[sink] = prefac
   
    # Only interested in mantle melting and crustal volcanic emplacement here. We'll let mantle emplaced 
    # melts do their trippy mixy thing. 
    #12 km == Y=0.981
    # 7km = Y = 0.99
    # 20 km = Y=0.969
    dT1[(Y > 0.981) & (dT2 > 0)] = 0.0  # No depletion of shallow crust, if melting
    dT1[(Y > 0.99)] = 0.0            # No depletion cooling of very shallow crust at all
    dT2[(Y<0.969) & (dT1 < 0)] = 0.0  # No melt heating of mantle, if it is depleting
 

    g = (1.0 - omegaT)*g + omegaT*geq + dt*0.5*H  + dt*0.5*dT1 + dt*0.5*dT2 - Sink  #scale for this term is numerical.


    #Streaming
    g[2,:,:] = np.roll(g[2,:,:],1,axis=1)
    g[6,:,:] = np.roll(g[6,:,:],-1,axis=0)
    g[6,:,:] = np.roll(g[6,:,:],1,axis=1)
    g[1,:,:]=np.roll(g[1,:,:],1,axis=0)
    g[5,:,:]=np.roll(g[5,:,:],1,axis=0)
    g[5,:,:]=np.roll(g[5,:,:],1,axis=1)
    
    g[4,:,:] = np.roll(g[4,:,:],-1,axis=1)
    g[8,:,:]= np.roll(g[8,:,:],1,axis=0)
    g[8,:,:]= np.roll(g[8,:,:],-1,axis=1)
  
    g[3,:,:] = np.roll(g[3,:,:],-1,axis=0)
    g[7,:,:] = np.roll(g[7,:,:],-1,axis=0)
    g[7,:,:] = np.roll(g[7,:,:],-1,axis=1)
    
    # BCs - As per working Snowy Hydro file
    # Left  i=0, adiabatic  
    g[:,0,:] = g[:,1,:]
    #LHS - constant temperature as per symmetric rift
    #g[6,0,:m] = w[6]*T_base + w[8]*T_base - g[8,0,:m]
    #g[3,0,:m] = w[3]*T_base + w[1]*T_base - g[1,0,:m]
    #g[7,0,:m] = w[7]*T_base + w[5]*T_base - g[5,0,:m]

    # Right,i=n, adiabatic 
    g[:,n,:] = g[:,n-1,:]
    # BOTTOM: j=m (inverting top and bottom definitions)
    g[6,:,0] = w[6]*T_base + w[8]*T_base - g[8,:,0]
    g[5,:,0] = w[5]*T_base + w[7]*T_base - g[7,:,0]
    g[2,:,0] = w[2]*T_base + w[4]*T_base - g[4,:,0]
    # TOP j=0  
    g[7,:,m] = w[7]*T_top + w[5]*T_top - g[5,:,m]
    g[4,:,m] = w[4]*T_top + w[2]*T_top - g[2,:,m]
    g[8,:,m] = w[8]*T_top + w[6]*T_top - g[6,:,m]

    th = g[0,:,:] + g[1,:,:] + g[2,:,:] + g[3,:,:] + g[4,:,:] + g[5,:,:] + g[6,:,:]+g[7,:,:]+g[8,:,:]
    #print("\tgeq\t",  geq[0,0:10,10])
    #print("\tg\t",  g[0,0:10,10])
    #print("\tth\t",  th[0:10,10])
    if(np.any(np.isnan(th))):
       print("NAN in th")
   
    return th, g

def LB_D2Q9_C(dt,m,n,cx,cy,omegaC,w,th,vx,vy,fc,fceq,C,Volc,dens,rho, Depl):
    #Collision
    np.set_printoptions(precision=3)
    fceq[0,:,:] = w[0]*C*(1.0+3.0*(vx*cx[0]+ vy*cy[0]))
    fceq[1,:,:] = w[1]*C*(1.0+3.0*(vx*cx[1]+ vy*cy[1]))
    fceq[2,:,:] = w[2]*C*(1.0+3.0*(vx*cx[2]+ vy*cy[2]))
    fceq[3,:,:] = w[3]*C*(1.0+3.0*(vx*cx[3]+ vy*cy[3]))
    fceq[4,:,:] = w[4]*C*(1.0+3.0*(vx*cx[4]+ vy*cy[4]))
    fceq[5,:,:] = w[5]*C*(1.0+3.0*(vx*cx[5]+ vy*cy[5]))
    fceq[6,:,:] = w[6]*C*(1.0+3.0*(vx*cx[6]+ vy*cy[6]))
    fceq[7,:,:] = w[7]*C*(1.0+3.0*(vx*cx[7]+ vy*cy[7]))
    fceq[8,:,:] = w[8]*C*(1.0+3.0*(vx*cx[8]+ vy*cy[8]))


    # Chemical terms

    prefacC = 1e-4
    prefac2=0.008 # This are equivalent to efficiency terms, determined heuristically.
    prefac1=0.008 # They have a minor effect against large excursions
    Volc_add = prefac1 * Volc / (dt)
    Depl_loss = prefac2 * Depl/(dt)
    #Depl_loss[Volc > 0] = 0.0  # These conditionals might drive depletion to +ve values
    #Depl_loss[Y > 0.981] = 0.0
    sinkC = C >1.5
    SinkC = np.zeros_like(C)
    SinkC[sinkC] = prefacC 
    sourceC = C <-1
    SourceC = np.zeros_like(C)
    SourceC[sourceC]=prefacC 

    fc = (1.0 - omegaC)*fc + omegaC*fceq +  dt*0.5*Volc_add - dt*0.5*Depl_loss - SinkC + SourceC
    
    #Streaming
    fc[2,:,:] = np.roll(fc[2,:,:],1,axis=1)
    fc[6,:,:] = np.roll(fc[6,:,:],-1,axis=0)
    fc[6,:,:] = np.roll(fc[6,:,:],1,axis=1)
    fc[1,:,:] = np.roll(fc[1,:,:],1,axis=0)
    fc[5,:,:] = np.roll(fc[5,:,:],1,axis=0)
    fc[5,:,:] = np.roll(fc[5,:,:],1,axis=1)
    
    fc[4,:,:] = np.roll(fc[4,:,:],-1,axis=1)
    fc[8,:,:] = np.roll(fc[8,:,:],1,axis=0)
    fc[8,:,:] = np.roll(fc[8,:,:],-1,axis=1)
  
    fc[3,:,:] = np.roll(fc[3,:,:],-1,axis=0)
    fc[7,:,:] = np.roll(fc[7,:,:],-1,axis=0)
    fc[7,:,:] = np.roll(fc[7,:,:],-1,axis=1)
    
    # BCs - As per working Snowy Hydro file
    # Left  i=0, adiabatic  
    #g[:,0,:] = g[:,1,:]
    #LHS - C adiabatic
    fc[:,0,:] = fc[:,1,:]
    # Right,i=n, adiabatic 
    fc[:,n,:] = fc[:,n-1,:]
    # BOTTOM: j=m (inverting top and bottom definitions)
    fc[:,:,0] =  fc[:,:,1] 
    # TOP j=0  
    fc[:,:,m] =  fc[:,:,m-1]

    C = fc[0,:,:] + fc[1,:,:] + fc[2,:,:] + fc[3,:,:] + fc[4,:,:] + fc[5,:,:] + fc[6,:,:]+fc[7,:,:]+fc[8,:,:]

    
    # Update density field dens THIS IS DIMENSIONALISED!
    density_S = [3200,3400, 3050, 2900]
    C_S = [-1, 0, 0.6, 1.0]
    fii = interp1d(C_S,density_S,fill_value=(3200,2900),bounds_error=False)
    dens = fii(C)
    #Eclogite
    f3 = (((Y< 0.95)&(Y>0.9))&( (C>0.7) ) )  
    #dens[f3] = 4092 - 500/(1+ 10**(Y[f3]*2)) #3300 #eclogite
    dens[f3] =  2900 + (3300-2900)/(1+ 10**(( (Y[f3] - 0.925)/(0.95-0.9) )*5.0))
    dens[ (Y<=0.9)&((C>0.7)) ] = 3300 # Full eclogite
    if(np.any(np.isnan(C))):
       print("NAN in C")
    if(np.any(np.isnan(dens))):
       print("NAN in dens")


    return C, dens, fc


def export_pic(l,th,vx,vy,X,Y,XL,YL,C,dens,Volc,Depl,melt_frac,shear_strain,visc,dir_out):
    plt.figure(figsize=(14,6))
    plt.rcParams["figure.dpi"]=300

    vxL = griddata((X.ravel(),Y.ravel()), vx.ravel(), (XL.ravel(),YL.ravel()), method='cubic')
    vyL = griddata((X.ravel(),Y.ravel()), vy.ravel(), (XL.ravel(),YL.ravel()), method='cubic')
    vxL = vxL.reshape(XL.shape)
    vyL = vyL.reshape(XL.shape)

    vmag =np.sqrt(vxL*vxL + vyL*vyL)
    conL = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    vmagH =np.sqrt(vx*vx + vy*vy)

    plt.clf()
    plt.figure(figsize=(14,6))
    plt.rcParams["figure.dpi"]=300
    title="timestep = "+str(l)

    ax1=plt.subplot(3,2,1)
    con1 = ax1.contour(X,Y,th,5,colors='black',alpha=0.3,linewidth=0.2)
    ax1.clabel(con1,inline=True,fontsize=8)
    ax1.set_ylabel("Y")
    ax1.set_xlabel("X")
    ax1.set_title(title)
    im=ax1.imshow(th.T, extent=[0, n*dx, 0, m*dy], origin='lower', cmap='inferno', alpha=0.5)
    div1=make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="5%", pad=0.5)
    cbar=plt.colorbar(im,cax=cax1,label='T',fraction=0.046, pad=0.04);
    cax1.invert_yaxis() 
    #ax1.set_aspect('auto')
    q1 = ax1.quiver(XL, YL, vxL, vyL, vmag, alpha=1.,cmap = 'magma_r')

    ax2=plt.subplot(3,2,2)
    ax2.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_title(title)
    #im2=ax2.imshow(vmagH.T, extent=[0, n*dx, 0, m*dy], origin='lower', cmap='RdGy_r', alpha=0.5)
    #im2=ax2.imshow(Volc.T, extent=[0, n*dx, 0, m*dy], origin='lower', cmap='RdGy_r', alpha=0.5)
    conL = np.array([0.1,0.2,0.4,0.6,0.8,1.0,1.2])
    cmap1=colormaps.get_cmap('RdGy_r')
    cmap2 = colormaps.get_cmap('cool_r')
    cmap3 = colormaps.get_cmap('plasma')

    cmap2._init() # create the _lut array, with rgba values
    cmap3._init()
    alphas = np.linspace(0, 0.8, cmap2.N+3)
    alphas2 = np.linspace(0, 0.8, cmap3.N+3)

    cmap2._lut[:,-1] = alphas
    cmap3._lut[:,-1] = alphas2

    img2 = ax2.imshow(Depl.T,  interpolation='nearest', extent=[0, n*dx, 0, m*dy], cmap=cmap1, origin='lower',alpha=1.0)
    img3 = ax2.imshow(Volc.T,  interpolation='nearest', extent=[0, n*dx, 0, m*dy], cmap=cmap2, origin='lower',vmin=0.0,vmax=np.mean(0)+2*np.std(Volc))
    img4  = ax2.imshow(melt_frac.T,  interpolation='nearest', extent=[0, n*dx, 0, m*dy], cmap=cmap3, origin='lower')
    #con1 = ax2.contour(X,Y,th,conL,colors='black',alpha=0.3,linewidth=0.3)
    #ax2.clabel(con1,inline=True,fontsize=8)

    div2=make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.5)
    cbar2=plt.colorbar(img2,cax=cax2,label='Depl/(Volc)',fraction=0.026, pad=0.02);
    #cbar2a=plt.colorbar(img3,cax=cax2,label='Volc',fraction=0.026, pad=0.02);

    #q1 = ax2.quiver(XL, YL, vxL, vyL, vmag, alpha=1.,cmap = 'magma_r')

    ax3=plt.subplot(3,2,3)
    ax3.set_ylabel("Y")
    ax3.set_xlabel("X")
    ax3.set_title(title)
    if (np.max(C) > 2000):
        Clim1 = 2
    else:
        Clim1=np.max(C)
    if (np.min(C) < -20000):
        Clim2 = -2
    else:
        Clim2=np.min(C)
    conL = np.array([-1.0,-0.5,-0.1,-0.05, 0.0, 0.05, 0.1,0.25,0.5,1.0])
    #con3 = ax3.contour(X,Y,C,conL,colors='black',alpha=0.3,linewidth=0.5)
    #ax3.clabel(con3,inline=True,fontsize=8)

    if (Clim1 >= 2000):
        im3 = ax3.imshow(C.T, extent=[0, n*dx, 0, m*dy], origin='lower', cmap='magma', alpha=0.99,vmin=Clim2,vmax=Clim1)
    else:
        im3 = ax3.imshow(C.T, extent=[0, n*dx, 0, m*dy], origin='lower', cmap='magma', alpha=0.99,vmin=np.mean(0)-2*np.std(C),vmax=np.mean(0)+2*np.std(C))
    div3=make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.5)
    cbar3=plt.colorbar(im3,cax=cax3,label='C',fraction=0.046, pad=0.04);
    #q1 = plt.quiver(XL, YL, vxL, vyL, vmag, alpha=1.,cmap = 'magma_r')

    ax4=plt.subplot(3,2,4)
    ax4.set_ylabel("Y")
    ax4.set_xlabel("X")
    ax4.set_title(title)
    fmax=3400
    fmin=2600
    #print("Dens",dens[100,100])
    #import seaborn as sns
    # cmap =  sns.diverging_palette(0,5,s=75,l=50,center="dark",as_cmap=True)
 
    #con4 = ax4.contour(X,Y,dens,10,colors='black',alpha=0.3,linewidth=0.5)
    #ax4.clabel(con4,inline=True,fontsize=8)
    im4=ax4.imshow(dens.T, extent=[0, n*dx, 0, m*dy], origin='lower', cmap='viridis', alpha=1.0,vmin=fmin,vmax=fmax)
    #cbar=plt.colorbar(label='Rho',fraction=0.046, pad=0.04);
    div4=make_axes_locatable(ax4)
    cax4 = div4.append_axes("right", size="5%", pad=0.5)
    cbar4=plt.colorbar(im4,cax=cax4,label='Density',fraction=0.046, pad=0.04)
    #cax4.get_xaxis().get_major_formatter().set_useOffset(False)
    #cbar4.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(np.min(rho), np.max(rho),200)], fontsize=10, weight='normal')
    cbar4.ax.set_yticklabels(["{:.3f}".format(i) for i in cbar4.get_ticks()])


    ax5=plt.subplot(3,2,5)
    ax5.set_ylabel("Y")
    ax5.set_xlabel("X")
    ax5.set_title(title)
    im5 = ax5.imshow(np.log10(shear_strain.T), extent=[0, n*dx, 0, m*dy], origin='lower', cmap='magma', alpha=0.99)

    div5=make_axes_locatable(ax5)
    cax5 = div5.append_axes("right", size="5%", pad=0.5)
    cbar5=plt.colorbar(im5,cax=cax5,label='log SS',fraction=0.046, pad=0.04);

    ax6=plt.subplot(3,2,6)
    ax6.set_ylabel("Y")
    ax6.set_xlabel("X")
    ax6.set_title(title)
    im6 = ax6.imshow(np.log10(visc.T), extent=[0, n*dx, 0, m*dy], origin='lower', cmap='Spectral', alpha=0.99)
    div6=make_axes_locatable(ax6)
    cax6 = div6.append_axes("right", size="5%", pad=0.5)
    cbar6=plt.colorbar(im6,cax=cax6,label='log Visc',fraction=0.046, pad=0.04);


    plt.tight_layout()

    str1=str("{:05.0f}".format(l))
    filenm = dir_out+"/impact23_"+str1+".png"
    plt.tight_layout()
    plt.savefig(filenm)
    plt.close("all")


def solve_viscosity(Y, th, omegaV, visc, fneq, high_stress_visc, N, yield_stress,rho,cx,cy, T_top,T_base,grav,visc_b,visc_cut,visc_scale):
    # Update T-dep viscosity
    T_ave = (T_top + T_base)/2.0
    tmp_th = np.copy(th)
    tmp_th[tmp_th < T_top] = T_top
    tmp_th[tmp_th > T_base] = T_base
    visc2 = np.exp(visc_b * (T_ave/tmp_th - 1)/(T_ave/T_top - 1))
    v0 = (visc_scale/np.min(visc2))
    visc2 = visc2 * v0
    visc_T = visc2
    visc_T[visc_T>visc_cut] = visc_cut
    

    # Plasticity
    S_prefac = 3 * omegaV/(2 * rho)
    # Assuming i,j are x,y directions:
    # 4x4 matrice
    #S_sum = fneq[0,:,:]*cx[0]*cy[0] + fneq[1,:,:]*cx[1]*cy[1] + fneq[2,:,:]*cx[2]*cy[2] + fneq[3,:,:]*cx[3]*cy[3] + fneq[4,:,:]*cx[4]*cy[4] + fneq[5,:,:]*cx[5]*cy[5] + fneq[6,:,:]*cx[6]*cy[6] + fneq[7,:,:]*cx[7]*cy[7] + fneq[8,:,:]*cx[8]*cy[8]
    cxx = cx*cx
    cyy = cy*cy
    cxy = cx*cy
    Sxx =  fneq[0,:,:]*cxx[0] + fneq[1,:,:]*cxx[1] + fneq[2,:,:]*cxx[2] + fneq[3,:,:]*cxx[3] + fneq[4,:,:]*cxx[4] + fneq[5,:,:]*cxx[5] + fneq[6,:,:]*cxx[6] + fneq[7,:,:]*cxx[7] + fneq[8,:,:]*cxx[8]
    Syy =  fneq[0,:,:]*cyy[0] + fneq[1,:,:]*cyy[1] + fneq[2,:,:]*cyy[2] + fneq[3,:,:]*cyy[3] + fneq[4,:,:]*cyy[4] + fneq[5,:,:]*cyy[5] + fneq[6,:,:]*cyy[6] + fneq[7,:,:]*cyy[7] + fneq[8,:,:]*cyy[8]
    Sxy =  fneq[0,:,:]*cxy[0] + fneq[1,:,:]*cxy[1] + fneq[2,:,:]*cxy[2] + fneq[3,:,:]*cxy[3] + fneq[4,:,:]*cxy[4] + fneq[5,:,:]*cxy[5] + fneq[6,:,:]*cxy[6] + fneq[7,:,:]*cxy[7] + fneq[8,:,:]*cxy[8]

    # 2nd invariant (I2) in 2D [a,b;c,d] = ad - bc
    #shear_strain = (Sxx*Syy - Sxy*Sxy)* S_prefac
    I1 = Sxx + Syy #1st stress invariant
    I2 = Sxx*Syy - Sxy*Sxy #2nd stress invariant
    J2 =  abs( ((I1**2)/3) - I2)  # invariant of the deviatoric stress tensor
    eff_stress = np.sqrt( 3 * J2)
    shear_strain = abs(eff_stress * S_prefac)
    
    #Variants:
    # J2 = 1/6( (Sxx-Syy)**2 +Sxx**2 +Syy**2 ) + Sxy**2
    #I2 = 0.5*(Sxx*Sxx + Syy*Syy) + Sxy*Sxy 


    #Sij = S_prefac * S_sum
    #shear_strain = (Sij * Sij)**0.5
    yield_stress = 0.1 * visc_T * 1e-3  + 0.01*grav*rho*(1 - Y) #np.max(shear_strain)
    v_plastic = np.maximum( high_stress_visc*np.ones_like(visc), (yield_stress/shear_strain)**N )
    stress = visc_T * shear_strain
    filt = (stress >= yield_stress)
    #filt = (J2 >= yield_stress)
    visc[filt] = v_plastic[filt]
    visc[~filt] = visc_T[~filt]
    #visc = 1/(1/visc_T + 1/v_plastic)

    #visc = visc_T
    #visc=visc_scale * np.ones_like(visc_T)
    omegaV=1.0/(3.*visc+0.5)

    return(visc,omegaV,shear_strain)



##################################################################
# Extemely slow - too many loops?
##################################################################


def run_loop(mstep,dt,m,n,gbeta,grav,cx,cy,omegaV,w,rho,th,C,vx,vy,f,feq,omegaT,T_top,T_base,g,geq,fc,fceq,H,alpha,X,Y,u0,dens,dir_out,Cp,im_file,visc,visc_b,visc_cut,visc_scale,high_stress_visc, N, yield_stress):
    nn = int(n/2)
    mm = int(m/2)
    l=0
    ll=0
    #fopen = open('Nu_t.csv','w')
    #np.savetxt(fopen,[],header="# t  Nu \n",newline='\n')
    #fopen.close()
    #Volc=0.0
    melt_dens = 2900
    melt_frac = np.zeros_like(th)
    Volc  = np.zeros_like(th)
    Depl  = np.zeros_like(th) 
    Tmagma  = np.zeros_like(th)
    k_impact = 0
    for t in range(mstep):
        # Melt injection stuff
        #x_Ts=3.0
        #x_Tl=4.0
        #print("2. dens",dens[nn,m])

        #Volc, Tmagma = magma_source_sol(th,C,Y,x_Ts,x_Tl,dx,dt,dens,melt_dens)
        # New column-by-column melt calculation - same as 1D now.
        if (im_file != None):
            start_melt_t = 2 * 500
            if t >= start_melt_t:
                melt_frac, Volc, Depl, Tmagma = loop_over_melt_columns(th,C,Y,x_Ts,x_Tl,dx,dt,dens,melt_dens)

            # Impact times
            # Note using LB scaling Kt/d^2 = K_lb*t_lb/d_lb^2
            # t.1e-6/(660e3)**2 = 0.000767*1/300**2 --> tdim = 117yrs, 500 timesteps === 50.5 kyrs (2117 images == 123 Myrs)
            im_ages = (np.genfromtxt(im_file,usecols=0)-400)*1e6 # Myrs -> yrs
            im_ages2 = np.genfromtxt(im_file,usecols=0)
            im_ages /= (117) # yrs to dt_lb
            # Impact image times are around [ 28.55555556,  55.5025641 ,  61.85811966,  74.04273504,
            #                                   104.51794872, 107.32649573, 129.46495726, 176.86666667]
            start_impact_t = 2*499
            im_ages -= im_ages[0]
            im_ages += start_impact_t
            if ( (t >= start_impact_t)&(t<im_ages[-1]) ):
                if (t > im_ages[k_impact]):
                    print("BOOM!!!",t,im_ages2[k_impact],im_ages[k_impact],k_impact)
                    th, g = impact_time(X,Y,th,g,Cp,m,im_file,k_impact)
                    visc, omegaV, shear_strain = solve_viscosity(Y, th, omegaV, visc, fneq, high_stress_visc, N, yield_stress,rho,cx,cy,  T_top,T_base,grav, visc_b,visc_cut,visc_scale)
                    export_pic(ll+0.5,th,vx,vy,X,Y,XL,YL,C,dens,Volc,Depl,melt_frac,shear_strain,visc,dir_out)
                    k_impact +=1

        rho, vx, vy, f, fneq = LB_D2Q9_V(dt,m,n,gbeta,cx,cy,omegaV,w,rho,th,vx,vy,f,feq,u0,Volc,Tmagma)
        th, g = LB_D2Q9_T(dt,m,n,cx,cy,omegaT,w,T_top,T_base,th,vx,vy,g,geq,H,Volc,Tmagma,x_Ts,Depl,Cp,Y)
        C, dens, fc = LB_D2Q9_C(dt,m,n,cx,cy,omegaC,w,th,vx,vy,fc,fceq,C,Volc,dens,rho,Depl)
        visc, omegaV, shear_strain = solve_viscosity(Y, th, omegaV, visc, fneq, high_stress_visc, N, yield_stress,rho,cx,cy,  T_top,T_base,grav, visc_b,visc_cut,visc_scale)


        print("Timestep",t,np.min(visc),"/",np.max(visc)," ...",np.min(shear_strain),"/",np.max(shear_strain))
        #print(t,dens[nn,mm],C[nn,mm],Volc[nn,mm],Tmagma[nn,mm])
        if (l % 500==0): #was 5000
            export_pic(ll,th,vx,vy,X,Y,XL,YL,C,dens,Volc,Depl,melt_frac,shear_strain,visc,dir_out)
            np.savetxt(dir_out+"/"+str(ll)+".dat",np.c_[np.ravel(th),np.ravel(vx),np.ravel(vy),np.ravel(C),np.ravel(dens),np.ravel(Volc),np.ravel(Depl),np.ravel(melt_frac),np.ravel(visc),np.ravel(shear_strain)])
            print(ll,l)
            ll += 1
            #calc_Nu(X,Y,th,T_top,T_base,alpha,m,t)
        l += 1
    return th, vx, vy, rho, C


def calc_Nu(X,Y,th,T_top,T_base,alpha,m,t):
    filt1 = ( ((Y> 0.9)&(Y<1.0)) & (th< 0.5) )
    dTdz =  np.mean(-1.0*(T_top - th[filt1])/(1.0 - Y[filt1])) #Assumed going up
    q_conv = alpha * dTdz  #alpha = k/rho*Cp  --> assume in lattice units rho=Cp=1
    q_cond = alpha * (T_base - T_top) # Note that m lattice units would appear in conv and cond and thus cancels out
    Nu = q_conv / q_cond
    fopen = open('Nu_t.csv','ab')
    np.savetxt(fopen, np.c_[float(t), float(Nu)],newline='\n')
    fopen.close()


def add_impact_heat(X,Y,xp,yp,th,g,R,vel,Cp,m):
    #Note R is in km, vel is in km/s
    # See Melosh (1989) or O'Neill et al. (2017) for terms
    # Tillotson constants
    C=7.24 #km/s
    S=1.25
    a=1.68
    b=2.74
    # Escape vel
    if (vel < 11.0):
        vel = 11.0
    d = 0.305*R*vel**0.361
    rc = 0.451*R*vel**0.211 *1e3
    r = np.linspace(0, 5*R,50)*1e3 #m
    u = (vel/2) *1e3 #m/s
    Ps = np.zeros_like(r)
    k=0
    for r2 in r:
        Ps[k] =  3400 * (C + S*u)*u
        if r2 > rc:
            Ps[k] =  Ps[0]*( (rc/r2)**(-a+b*np.log10(vel)) )
        k+=1
    #print(i, age[i]," Ma ","R",R[i], " rc",rc,"uc",u,"Ps",np.max(Ps)/1e9)
    f=(-2*S*Ps/(C*C*3400)) * ( 1.0 / (1 - np.sqrt( ((4*S*Ps)/(C*C*3400)) + 1)) )
    dT = (1/Cp) * ( (Ps/(C*C*3400))*(1 - (1/f) ) - ( (C**2/(S**2)) * (f - np.log(f) - 1)))
    # Reference to position, xp, yp
    rm = r * 1/660e3   #Scale to LB units, change for upper/whole mantle.
    rcm = rc *1/660e3
    xpm = 2 * ((xp)/12 ) # Long to L.U.
    ypm = d/660 # Emplacement depth is d
    f = interp1d(r,dT)
    r2 = np.sqrt( (X-xpm)**2 + ((1-Y) - ypm)**2)
    print("Scales - X/y:",np.max(X),np.max(Y),np.max(R),np.max(rm),np.max(r2),np.max(rcm),"Centre: (",xpm,ypm,")")
    
    filt1= (r2 < np.max(rm) )  #r2 is 2D, r is 1D
    dT2 = np.zeros_like(th)
    dT2[filt1] = f(r2[filt1]) #Interpolate dT to r2
    th += dT2
    filt1 = th > x_Tl
    th[filt1] = x_Tl[filt1]
    g[0,:,:] =  w[0]*th
    g[1,:,:] =  w[1]*th
    g[2,:,:] =  w[2]*th
    g[3,:,:] =  w[3]*th
    g[4,:,:] =  w[4]*th
    g[5,:,:] =  w[5]*th
    g[6,:,:] =  w[6]*th
    g[7,:,:] =  w[7]*th
    g[8,:,:] =  w[8]*th #
    return (th, g)

def impact_time(X,Y,th,g,Cp,m,im_file,k_impact):
    # Define xp, yp here from input long
    lon = np.genfromtxt(im_file,usecols=1)
    # From age list, get appropriate impact
    ages = np.genfromtxt(im_file,usecols=0)
    #i = np.argmin(np.abs(ages - t))
    xp = lon[k_impact]
    yp = 0.0

    R = np.genfromtxt(im_file,usecols=3)[k_impact]
    vel = np.genfromtxt(im_file,usecols=4)[k_impact]
    th, g = add_impact_heat(X,Y,xp,yp,th,g,R,vel,Cp,m)
    return (th, g)

#Geometry
m=300
n=600

dist_x = 2.0
dist_y = 1.0
# Backing out increments
dx = dist_x/n
dy= dist_y/m
dt=1.000  # This should have no effect except on H 

# Initiatialise grid
x = np.arange(0,dist_x+dx,dx) #Not n+1 ... mmm.
y = np.arange(0,dist_y+dy,dy)
#print(y)
X,Y = np.meshgrid(x,y) # Change rows to columns
X=X.T
Y=Y.T

f=np.zeros((9,n+1,m+1))
feq=np.zeros((9,n+1,m+1))

fc=np.zeros((9,n+1,m+1))
fceq=np.zeros((9,n+1,m+1))

g=np.zeros((9,n+1,m+1))
geq=np.zeros((9,n+1,m+1))

w = np.zeros((9))
cx = np.zeros((9))
cy = np.zeros((9))

u0="free" # for single value: 0.0 or for array: np.ones_like(x)*0.01
grav=0.1
beta=0.001

#This is dimensionalised in melt calcs - taken care of internally (rho < 4000 kg/m3). 
rho0=1.0  #Non-dim version
dens0 = 3400 #Dim version for melting

rho=np.ones((n+1,m+1))*rho0
dens=np.ones((n+1,m+1))*dens0

###################################################
#ICs
##################################################
# Vel conditions
if (type(u0) == float)or(type(u0) == np.array):
    vx=np.ones((n+1,m+1))*u0
else:
    vx=np.zeros((n+1,m+1))
vy=np.zeros((n+1,m+1))
th=np.ones((n+1,m+1))
C=np.ones((n+1,m+1))

# T conditions
T_top=0.01  # Avoids singularities in viscosity laws
T_base=1.01 # 0.001
T_ave = (T_top + T_base)/2.0


#Impose initial linear gradient on Temp field
#rge = (T_base - T_top)*0.0005

#T1 = T_base*np.ones(m+1)
C1 = np.linspace(0.00,0.0,m+1)
z1=np.linspace(660e3,0,m+1)
# Age of litho being thinned - 350 Myr here.
T1 = 0.9*erf((z1)/(2*np.sqrt(1e-6*(45*1e6*60*60*24*365.25))))
T2 = 0.9*erf((z1)/(2*np.sqrt(1e-6*(90*1e6*60*60*24*365.25))))
for j in np.arange(0,m+1,1):
    th[:,j] = th[:,j]*T2[j]
    C[:,j] = C[:,j]*C1[j]


T_ave = (np.min(th) + np.max(th))/2
rho = rho*(1 - beta*(th - T_ave))
# Apply initial crustal layer
#h=0.05
#C = 0.65*(1/(3.7612638903183755*h*np.sqrt(np.pi))) * np.exp(-((Y-1)/h)**2)
#(1/(h*np.sqrt(np.pi))) * np.exp(-((C-1)/h)**2)

#C[filt1]=1.0
#rho[filt1]=0.9

density_S = [3200,3400, 3050, 2900]
C_S = [-1, 0, 1, 1.5]
fii = interp1d(C_S,density_S,fill_value=(3200,2900),bounds_error=False)
dens = fii(C)

############################################################################################
# Bespoke thermal ICs

thermal_pulse = False
#Initial simple thermal/impact pulse
if thermal_pulse==True:
    R = np.sqrt((X-1.0)**2 + (Y-0.05)**2)
    filtR = (R < 0.2)&(Y<0.95)
    th[filtR] += 0.05
    R = np.sqrt((X-0.0)**2 + (Y-0.05)**2)
    filtR = (R < 0.2)&(Y<0.95)
    th[filtR] += 0.02
    R = np.sqrt((X-2.0)**2 + (Y-0.05)**2)
    filtR = (R < 0.2)&(Y<0.95)
    th[filtR] += 0.05

# Impact heating############################################################################
# See O'Neill et al. 2017 for further details and parameters.
#Initial thermal/impact pulse
# Imposing single impact here

impact_heat_IC=False

if impact_heat_IC==True:
    ND_T = 2000 #UM-LM non-adiabatic temperature

    R = np.sqrt((X-1.0)**2 + (Y-0.95)**2)
    filtR = (R < 0.75)&(Y<0.95)

    Ri = 100.0 #km
    vel = 26.0  # escape vel, km/s
    Ci=7.24 #km/s
    S=1.25
    a=1.68
    b=2.74
    Cp=840
    M2S = 1e6*365.25*24*60*60

    d = 0.305*Ri*vel**0.361
    rc = 0.451*Ri*vel**0.211 *1e3
    r = np.linspace(0, 20*Ri,75)*1e3 #m - for calculating shells of heating
    u = (vel/2) *1e3 #m/s

    Ps = np.zeros_like(r)
    k=0
    for r2 in r:
     Ps[k] =  3400 * (Ci + S*u)*u
     if r2 > rc:
         Ps[k] =  Ps[0]*( (rc/r2)**(-a+b*np.log10(vel)) )
     k+=1
    f_imp=(-2*S*Ps/(Ci*Ci*3400)) * ( 1.0 / (1 - np.sqrt( ((4*S*Ps)/(Ci*Ci*3400)) + 1)) )
    dT = (1/Cp) * ( (Ps/(Ci*Ci*3400))*(1 - (1/f_imp) ) - ( (Ci**2/(S**2)) * (f_imp - np.log(f_imp) - 1))) # function of r
    print(np.c_[r/(1e3*660),dT/ND_T])
    print("dT:",np.min(dT),np.max(dT))
    # Interpolate to X,Y grid
    f_imp=interp1d(r/(660*1e3),dT/ND_T,fill_value=(np.max(dT)/ND_T,0),bounds_error=False) #Non-Dim the relationship for grid interp
    dT_X = f_imp(np.ravel(R))
    dT_X = dT_X.reshape(X.shape)
    #np.set_printoptions(threshold=np.inf)
    print(np.c_[np.ravel(R[dT_X>0.1]),np.ravel(dT_X[dT_X>0.1])])
    #dT_X = dT_X.reshape(X.shape)
    print("dT_X:",np.min(dT_X),np.max(dT_X))

    #th += dT_X
    #th[filtR] += dT_X[filtR] # The example shown restricts the impact influence radially and prevents crustal disturbance. Remove filter to remove these effects. 
    th[th>1.0] = 1.0  # Cap to effective liquidus. 

# Solidus condition
#th[filtR] += 0.12 # x_Ts[filtR] -0.1
#th[th>1.0] = 1.0

#(co) Sinusoidal IC
#Tanom = 0.02*np.cos(np.linspace(0,2*np.pi,n+1))
#for i in np.arange(0,n+1,1):
#    #rho[i,:] += Tanom[i]
#    th[i,:] += Tanom[i]


# End ICs
############################################################################################################


print("Shapes",np.shape(X),np.shape(Y),np.shape(th))


# Initialise variables
sol = (np.genfromtxt("data/stixrude_sol.txt",usecols=0) )
sol_x = (np.genfromtxt("data/stixrude_sol.txt",usecols=1))
sol = sol - 0.15*sol_x #Remove adiabatic gradient
sol = (sol - 273.0 ) /2000.0 # Scale
sol_x = sol_x*1e3 / 660e3

fi=interp1d(sol_x,sol,fill_value="extrapolate")
#Y2 = np.flipud(Y)
Y2 = Y*(-1.0) + 1.0 # Checked.
x_Ts = fi(Y2)
#print("x_Ts_2:",np.shape(x_Ts),x_Ts.dtype)
x_Tl = x_Ts + 0.3 #assumed liquidus is just 300C above solidus for now.

w=np.zeros(9)
w[0]=4./9.
w[1]=w[2]=w[3]=w[4]=1./9.
w[5]=w[6]=w[7]=w[8]=1./36.

cx = np.array([0.0,1.0,0.0,-1.0,0.0,1.0,-1.0,-1.0,1.0])
cy = np.array([0.0,0.0,1.0,0.0,-1.0,1.0,1.0,-1.0,-1.0])


# S' = d'**2/( K' * T'),  d'=4e6/(0.00130 * 2000) & / Cp(=1000) for LB sources
# H_mantle_nonchron = 3.2468e-12 W/kg
Hm = 4.9797e-9 # + 5.6135e-8
# H OIB @ 4Ga = 9.008e-10
Hc = 1.38159e-6 # + 8.1288e-11 #latter values are impact heating
H=np.ones_like(th)* Hm
H[Y>0.99]= Hc  # 20km thick OIB crust.... Pretty hot.

Cp=1.0
gbeta = grav*beta

ra=5e6 #4.167e4 # Might be slightly subcritical if ~1e4
pr=1e3 #Mora 23


##################################################################################################
# T-dep viscosity. Here we initialise omega matrix, and update it after the temperature step
#

visco2 = (gbeta*(T_base - T_top) * (m-1)**3 )/(ra * 1/pr)  #Note density is removed from this, so assumed non-dim here
visc_scale = np.sqrt(visco2)
R=1
E=8.5 #5.5
# Mora formulation
visc_b = 5*E/T_ave
visc2 = np.exp(visc_b * (T_ave/th - 1)/(T_ave/T_top - 1))
v0 = (visc_scale/np.min(visc2))
visc2 = visc2 * v0
visc = visc2
visc_cut=5e5
visc[visc>visc_cut] = visc_cut
omegaV=1.0/(3.*visc+0.5)

high_stress_visc = 0.01 * visc_scale
N =  1.0
shear_strain = 1.0e-5
yield_stress = 0.8 * visc_scale * np.max(shear_strain) # Note shear strain is not calculated till within the loop. Need a priori info for this. Start high and rely on updating

#visco=0.1 #0.0405  # Modified to slow down convection - CO 13/Apr/2023

##################################################################################################


#alpha=visco/pr  #Their alpha ~ thermal diffusivity. I think this is too small. 
alpha=visco2/pr
Cdiff=6e-5 #Compositional diffusion

print("Ra=",ra,"gbeta:",gbeta)
omegaT=1.0/(3.*alpha+0.5)
omegaC=1.0*np.ones_like(C)/(3*Cdiff+0.5) #assumed dt and csq are 1... omegaC=1.0/(3*Cdiff/(dt*csq)+0.5)
# Increase diff to counter corner instabilities
filtC1 = (Y>0.9)&(X<0.1)
omegaC[filtC1] = 1.0/(3*18*Cdiff+0.5)
filtC2 = (Y>0.9)&(X>1.9)
omegaC[filtC2] = 1.0/(3*18*Cdiff+0.5)

mstep=2500000 #150000
#print("Omegas",omegaV,omegaT,omegaC,alpha,visco)


dir_out=str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
os.makedirs(dir_out, exist_ok=True)
file_out = dir_out+"/input.py"
shutil.copy(__file__, file_out)
file_2 = dir_out+"/XY.dat"
np.savetxt(file_2,np.c_[np.ravel(X),np.ravel(Y)])

# Initial vels
Volc = np.zeros_like(th)
# Low resolution for plotting velocity field
XL,YL = np.meshgrid(np.linspace(0,2,40),np.linspace(0,1,20))
#Run main

feq = loopy_loop(n,m,vx,vy,cx,cy,rho,w,feq)

geq[0,:,:] = w[0]*th*(1.0+3.0*(vx*cx[0]+ vy*cy[0]))
geq[1,:,:] = w[1]*th*(1.0+3.0*(vx*cx[1]+ vy*cy[1]))
geq[2,:,:] = w[2]*th*(1.0+3.0*(vx*cx[2]+ vy*cy[2]))
geq[3,:,:] = w[3]*th*(1.0+3.0*(vx*cx[3]+ vy*cy[3]))
geq[4,:,:] = w[4]*th*(1.0+3.0*(vx*cx[4]+ vy*cy[4]))
geq[5,:,:] = w[5]*th*(1.0+3.0*(vx*cx[5]+ vy*cy[5]))
geq[6,:,:] = w[6]*th*(1.0+3.0*(vx*cx[6]+ vy*cy[6]))
geq[7,:,:] = w[7]*th*(1.0+3.0*(vx*cx[7]+ vy*cy[7]))
geq[8,:,:] = w[8]*th*(1.0+3.0*(vx*cx[8]+ vy*cy[8])) 

fceq[0,:,:] = w[0]*C*(1.0+3.0*(vx*cx[0]+ vy*cy[0]))
fceq[1,:,:] = w[1]*C*(1.0+3.0*(vx*cx[1]+ vy*cy[1]))
fceq[2,:,:] = w[2]*C*(1.0+3.0*(vx*cx[2]+ vy*cy[2]))
fceq[3,:,:] = w[3]*C*(1.0+3.0*(vx*cx[3]+ vy*cy[3]))
fceq[4,:,:] = w[4]*C*(1.0+3.0*(vx*cx[4]+ vy*cy[4]))
fceq[5,:,:] = w[5]*C*(1.0+3.0*(vx*cx[5]+ vy*cy[5]))
fceq[6,:,:] = w[6]*C*(1.0+3.0*(vx*cx[6]+ vy*cy[6]))
fceq[7,:,:] = w[7]*C*(1.0+3.0*(vx*cx[7]+ vy*cy[7]))
fceq[8,:,:] = w[8]*C*(1.0+3.0*(vx*cx[8]+ vy*cy[8]))

f[0,:,:] =  w[0]*rho
f[1,:,:] =  w[1]*rho
f[2,:,:] =  w[2]*rho
f[3,:,:] =  w[3]*rho
f[4,:,:] =  w[4]*rho
f[5,:,:] =  w[5]*rho
f[6,:,:] =  w[6]*rho
f[7,:,:] =  w[7]*rho
f[8,:,:] =  w[8]*rho

g[0,:,:] =  w[0]*th
g[1,:,:] =  w[1]*th
g[2,:,:] =  w[2]*th
g[3,:,:] =  w[3]*th
g[4,:,:] =  w[4]*th
g[5,:,:] =  w[5]*th
g[6,:,:] =  w[6]*th
g[7,:,:] =  w[7]*th
g[8,:,:] =  w[8]*th

fc[0,:,:] =  w[0]*C
fc[1,:,:] =  w[1]*C
fc[2,:,:] =  w[2]*C
fc[3,:,:] =  w[3]*C
fc[4,:,:] =  w[4]*C
fc[5,:,:] =  w[5]*C
fc[6,:,:] =  w[6]*C
fc[7,:,:] =  w[7]*C
fc[8,:,:] =  w[8]*C


usum = f[0,:,:]*cx[0] + f[1,:,:]*cx[1] + f[2,:,:]*cx[2] + f[3,:,:]*cx[3] + f[4,:,:]*cx[4] +f[5,:,:]*cx[5] + f[6,:,:]*cx[6] +f[7,:,:]*cx[7]+f[8,:,:]*cx[8]
vsum = f[0,:,:]*cy[0] + f[1,:,:]*cy[1] + f[2,:,:]*cy[2] + f[3,:,:]*cy[3] + f[4,:,:]*cy[4] +f[5,:,:]*cy[5] + f[6,:,:]*cy[6] +f[7,:,:]*cy[7]+f[8,:,:]*cy[8]
vx = usum/rho
vy = vsum/rho

rho, vx, vy, f, fneq = LB_D2Q9_V(dt,m,n,gbeta,cx,cy,omegaV,w,rho,th,vx,vy,f,feq,u0, Volc, Volc)
th, g = LB_D2Q9_T(dt,m,n,cx,cy,omegaT,w,T_top,T_base,th,vx,vy,g,geq,H,Volc,0,x_Ts,Volc,Cp,Y)
C, dens, fc = LB_D2Q9_C(dt,m,n,cx,cy,omegaC,w,th,vx,vy,fc,fceq,C,Volc,dens,rho,0)        
visc, omegaV, shear_strain = solve_viscosity(Y, th, omegaV, visc, fneq, high_stress_visc, N, yield_stress,rho,cx,cy,  T_top,T_base,grav, visc_b,visc_cut,visc_scale)

vx=0.0*vx
vy=0.0*vy

# Impacts
#im_file = "impact_subset.dat"
im_file="data/Impacts_lon204.dat"
#im_file=None
th, vx, vy, rho = run_loop(mstep,dt,m,n,gbeta,grav,cx,cy,omegaV,w,rho,th,C,vx,vy,f,feq,omegaT,T_top,T_base,g,geq,fc,fceq,H,alpha,X,Y,u0,dens,dir_out,Cp,im_file,visc,visc_b,visc_cut,visc_scale,high_stress_visc, N, yield_stress)





