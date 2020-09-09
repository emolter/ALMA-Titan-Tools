#! /usr/bin/env python

import numpy as np
import matplotlib.pylab as plt
import matplotlib
from matplotlib import gridspec 
from pymodules.titan import titanSr
import glob

filename=glob.glob('*.mre')[0]

print("Reading .mre file "+filename)

f=open(filename)
filedata = f.readlines()
header = filedata[0:5]
data = filedata[5:5+int(header[1].split()[2])]

ndata=int(header[1].split()[2])
nvar=int(header[1].split()[3])

i = [];  wn =[];  model=[];err=[];perr=[]; spec=[]; pdiff=[];
for line in data: 
    a,b,c,d,e,f,g = line.split() 
    i.append(float(a))
    wn.append(float(b))
    spec.append(float(c))
    err.append(float(d))
    perr.append(float(e))
    model.append(float(f))
    pdiff.append(float(g))

fig = plt.figure(figsize=(10.5,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.0)

ax=plt.subplot(gs[0])
ax.set_title('Observed spectrum and NEMESIS model')
ax.get_xaxis().get_major_formatter().set_useOffset(False)
#ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
#ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
ax.tick_params(labelsize=18)
plt.setp(ax.get_xticklabels(), visible=False)
#ax.set_ylim(2.5,8.5)
ax.plot(wn,spec,'-k',drawstyle='steps-mid',label='Observation')
ax.plot(wn,model,label='Model')
ax.set_ylabel('Radiance (nW/m$^2$/sr/cm$^{-1}$)')


ax2=plt.subplot(gs[1],sharex=ax)
#ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
ax2.plot(wn,np.asarray(spec)-np.asarray(model),'-k',drawstyle='steps-mid')
ax2.plot(wn,np.zeros(ndata),'-r')

ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
ax2.set_ylabel('Residuals')
ax2.tick_params(labelsize=18)
ax2.set_xlim(np.min(wn),np.max(wn))
#ax2.set_ylim(-0.3,0.3)

chisq = np.sum((np.asarray(spec)-np.asarray(model))**2/np.asarray(err)**2)
DOF = len(spec)-nvar
print("Nvar = ",nvar)
print("Chisq_R =",chisq / DOF)


plt.show()

