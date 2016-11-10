'''Plots Rich's temperature fields as a pretty picture of Titan's disk.  Includes sub observer latitude and angle. '''

import numpy as np
import matplotlib.pyplot as plt
from pylab import colorbar
from scipy.interpolate import interp1d

######################################################
##################### INPUTS #########################
######################################################

infile = 'T111.t.zonal.dat' #one of Rich's temperature map files
outfigure = 'tmap_T111_alt200.ps'
Titan_radius = 2575.0 #km
top_atm = 1200.0
subobslat = 21.18 #sub-observer latitude (degrees)
ccw = 20.0 #rotation angle of planet counterclockwise from sky north (degrees)
dx = 5 #resolution of model Titan in km
map_alt = 200 #altitude where you want to map in km
extra_size = 50 #buffer zone around figure to make it look nice

def temp_read(infile):
    '''Temperature field data from Rich Achterberg in form latitude, pressure, temperature, ?, ?, ?, altitude, write as numpy array'''
    tfield = []
    with open(infile,'r') as f:
        for line in f:
            l = line.split()
            l = [float(val.strip(', \n')) for val in l]
            tfield.append(l)
    return np.asarray(tfield).T

def tmap_extract(infile,standard_h):
    '''Get temperature profile from file into a useful format: list of latitudes, vertical profile for each latitude.  Vertical profiles are interpolated onto a standard height grid so that tempWts can average them'''
    #Extract latitudes
    tfield = temp_read(infile).tolist()
    lats, indices, counts = np.unique(tfield[0], return_index=True, return_counts=True)

    #Extract temperature profiles at each latitude, interpolate onto a standard altitude grid
    temps = []
    for j in range(len(lats)):
        #lat = float(lats[j])
        p = tfield[1][indices[j]:indices[j]+counts[j]]
        t = np.asarray(tfield[2][indices[j]:indices[j]+counts[j]])
        h = np.asarray(tfield[-1][indices[j]:indices[j]+counts[j]])

        with np.errstate(divide='ignore',invalid='ignore'): #interp1d doesn't love nan values, but it doesn't break anything to keep them in there
            func = interp1d(h,t,kind='linear',bounds_error = False)
            t_interp = func(standard_h)
        temps.append(t_interp.tolist())

    temps = np.asarray(temps).T.tolist()
    return (temps,lats)

def tmap(ts,lats,Titan_radius,subobslat,ccw,dx,map_h,extra_size):
    subobslat = np.deg2rad(subobslat)
    ccw = np.deg2rad(ccw)
    ht_i = int(np.where(standard_h == map_h)[0])

    #Finding vector of true north of Ttan
    northx = -np.sin(ccw)*np.cos(subobslat)
    northy = np.cos(ccw)*np.cos(subobslat)
    northz = np.sin(subobslat) 
    
    #create grid z where each point represents Titan latitude
    lim = Titan_radius + map_h + extra_size #add extra size here so map extends to look nicer
    xx = slice(-lim,lim+dx,dx)
    y,x = np.mgrid[xx,xx]
    z = y*x*0.0 #initializes array z, which becomes the latitudes on the 2-d Titan map - confusing I know
    with np.errstate(divide='ignore',invalid='ignore'): #We actually want all y > Titan_radius + top_atm to be nans, so the invalid inputs to arcsin are helping here
        zcoord = np.sqrt((Titan_radius+map_h)**2 - x**2 - y**2) #these are the actual z-coordinates (distance from Titan center to observer) at each x,y point
        dprod = (northx*x + northy*y + northz*zcoord)/(Titan_radius+map_h) #dot product of north pole vector and each vector in model planet
        z = 90 - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid
        for j in range(1,len(lats)):
            z[np.logical_and(z >= lats[j-1], z < lats[j])] = lats[j-1] #discretize z into the latitudes given in the input data
            z[z == lats[j-1]] = ts[ht_i][j-1] #turn those latitudes into their corresponding temperatures at a given altitude

    #Plot
    fig,ax = plt.subplots(figsize = (15,15))
    cmap = plt.cm.coolwarm
    #cmap.set_bad(color='grey')
    cax = ax.imshow(z, cmap=cmap, origin='lower')
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('Temperature',fontsize=18)
    plt.title('CIRS Temperature Map: Altitude '+str(map_h)+' km',fontsize=20)
    plt.show()
    fig.savefig(outfigure,bbox = 'None')
    return

standard_h = np.arange(0.0,540.0,20.0)
map_h = map_alt - map_alt % 20

(ts,lats) = tmap_extract(infile,standard_h)
tmap(ts,lats,Titan_radius,subobslat,ccw,dx,map_h,extra_size)
