import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

### Ned Molter 06/24/16 ###

######################################################
##################### INPUTS #########################
######################################################

#Filenames
infile = 'T111.t.zonal.dat' #one of Rich's temperature map files
img = 'spw0.clean.image' #Only used if usecasa = True
outfile = 'T111.txt'
outfigure = 'T111.ps'

#Options
usecasa = False #True if you're extracting the beam shape with CASA, False if you're defining the Gaussian beam manually
diagnostics = False #look at plots of beam and model Titan
diskavg = False #if true, sets cx = cy = [0.0] and computes weights for whole planet instead of convolving with Gaussian

#Planet parameters
Titan_dist = 9.08165474456227 #au
Titan_radius = 2575.0 #km
subobslat = 21.25 #sub-observer latitude (degrees)
ccw = 0.6898 #rotation angle of planet counterclockwise from sky north (degrees)
if not usecasa:
    beamx = 0.1 #arcsec
    beamy = 0.08 #arcsec
    theta = 20.0 #degrees

#Model parameters
cx = [0.0]
cy = [-2500] #cx,cy is a grid of coordinates defining of center-of-beam positions (in km from the center of the planet) at which an average temperature is retrieved
standard_h = np.arange(40.0,540.0,20.0) #height grid over which to compute model (min,max,step). Note for each altitude step the value in here takes the place of top_atm in tempWts
dx = 10 #resolution of model Titan in km. Runtime of code goes as about dx^-2, so degrading resolution helps speed things up. Note there are about 7 km per milliarcsec at Titan's distance







######################################################
############### FUNCTION DEFINITIONS #################
######################################################

def temp_read(infile):
    '''Temperature field data from Rich Achterberg in form latitude, pressure, temperature, ?, ?, ?, altitude, write as numpy array'''
    tfield = []
    with open(infile,'r') as f:
        for line in f:
            l = line.split()
            l = [float(val.strip(', \n')) for val in l]
            tfield.append(l)
    return np.asarray(tfield).T


def casa_extract(img):
    '''Brings image header info into script using imhead'''
    cubeheader = imhead(imagename=img,mode='list')
    (beamx,beamy) = (float(cubeheader['beammajor']['value']),float(cubeheader['beamminor']['value']))
    theta = float(cubeheader['beampa']['value']) #here theta defined as angle from north to east in degrees
    (refx,refy) = (float(cubeheader['crpix1']),float(cubeheader['crpix1'])) #center pixel
    (pixszx,pixszy) = (np.degrees(np.fabs(3600*float(cubeheader['cdelt1']))),np.degrees(np.fabs(3600*float(cubeheader['cdelt2']))))
    return [beamx,beamy,theta,refx,refy,pixszx,pixszy]

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

def gauss2d(x,y,fwhm_x,fwhm_y,x0,y0,theta):
    """ Takes in np arrays for x,y. Returns the value of the 2d elliptical gaussian. Theta defined as angle in radians from x axis to semimajor axis """
    sig_x = fwhm_x / (2*np.sqrt(2*np.log(2)))
    sig_y = fwhm_y / (2*np.sqrt(2*np.log(2)))
    A = 1.0
    a1 = np.cos(theta)**2/(2*sig_x**2) + np.sin(theta)**2/(2*sig_y**2)
    b1 = -np.sin(2*theta)/(4*sig_x**2) + np.sin(2*theta)/(4*sig_y**2)
    c1 = np.sin(theta)**2/(2*sig_x**2) + np.cos(theta)**2/(2*sig_y**2)
    g = A*np.exp(-(a1*(x-x0)**2 - 2*b1*(x-x0)*(y-y0) + c1*(y-y0)**2))
    return g


def tempWts(lats,cx,cy,beamx,beamy,theta,Titan_radius,Titan_dist,top_atm,dx,subobslat,ccw,diagnostics):
    """ Takes in definition of elliptical beam at arbitrary location on Titan's disk.  Returns weightings to be used in .spx header.
    cx,cy are grid of coords of center of beam in km from center of Titan in km. """

    Titan_dist = Titan_dist*149597870 #to km
    a = Titan_dist*np.radians(np.arcsin(beamx/3600))
    b = Titan_dist*np.radians(np.arcsin(beamy/3600))

    subobslat = np.deg2rad(subobslat)
    ccw = np.deg2rad(ccw)

    #Finding vector of true north of Ttan
    northx = -np.sin(ccw)*np.cos(subobslat)
    northy = np.cos(ccw)*np.cos(subobslat)
    northz = np.sin(subobslat) 
    
    #create grid z where each point represents Titan latitude
    lim = Titan_radius + top_atm + a #put a here so map extends out to a full beam past top of atmosphere
    xx = slice(-lim,lim+dx,dx)
    y,x = np.mgrid[xx,xx]
    z = y*x*0.0 #initializes array z, which becomes the latitudes on the 2-d Titan map - confusing I know
    with np.errstate(divide='ignore',invalid='ignore'): #We actually want all y > Titan_radius + top_atm to be nans, so the invalid inputs to arcsin are helping here
        zcoord = np.sqrt((Titan_radius+top_atm)**2 - x**2 - y**2) #these are the actual z-coordinates (distance from Titan center to observer) at each x,y point
        dprod = (northx*x + northy*y + northz*zcoord)/(Titan_radius+top_atm) #dot product of north pole vector and each vector in model planet
        z = 90 - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid
        for j in range(1,len(lats)):
            z[np.logical_and(z >= lats[j-1], z < lats[j])] = lats[j-1] #discretize z into the latitudes given in the input data

    #make grid rad where each point represents distance from center of Titan in km
    rad = np.sqrt(x**2 + y**2)

    #Handle bad and out-of-bounds values
    z[z == -1.0] = float('NaN')
    z[rad >= Titan_radius + top_atm] = float('NaN')
    
    #make gaussian
    g = gauss2d(x,y,a,b,cx,cy,theta)
    
    #compute normalized weights
    wts = {}
    for val in lats:
        garr = g[np.where(z == val)]
        wts[val] = sum(garr)
    # gnanarr =  g[np.where(np.isnan(z))] #treating NaN values such that weighting will not add up to one
    #s = sum(wts.values())+sum(gnanarr)
    s = sum(wts.values())
    for key,val in wts.items():
        val = float(val)/float(s)
        wts[key] = val

    meanlat = sum([val*key for key,val in wts.items()])
        
    ### PLOTS ###    
    #latitude map
    if diagnostics and k == 3:
        fig1 = plt.figure(figsize = (15,15))
        ax = fig1.add_subplot(111)
        ax.imshow(z, cmap='RdBu',origin='lower')
        plt.show()
        
    #Gaussian laid onto circular region
    if diagnostics and k == 3:
        fig3 = plt.figure(figsize = (15,15))
        ax = fig3.add_subplot(111)
        ax.imshow(np.multiply(g,z*0.0 + 1),cmap = 'RdBu',origin='lower')
        plt.show()
        
    ## #Gaussian convolved with latitude map
    ## if diagnostics and k == 3:
    ##     conv = np.multiply(g,z)
    ##     fig2 = plt.figure(figsize = (15,15))
    ##     ax = fig2.add_subplot(111)
    ##     ax.imshow(conv, cmap='RdBu',origin='lower')
    ##     plt.show()
        
    return wts, meanlat

def meanTemp(wts,lats,temps):
    '''Makes mean temperature, dealing with bad values by renormalizing weights
    '''
    tlist = []
    wtlist = []
    latlist = []
    noDataList = []
    #find bad values
    for i in range(len(lats)):
        lat = lats[i]
        wt = wts[lat]
        t = temps[i]        
        if t <= 0.0 or np.isnan(t):
            noDataList.append(lat)
        else:
            tlist.append(t)
            wtlist.append(wt)
            latlist.append(lat)

    if len(tlist) == 0: #if no data at this altitude
        return float('NaN')
    else:
        renorm = sum(wts.values())/sum(wtlist)

        #use good values to get a correct temperature average
        tave = 0.0
        for j in range(len(latlist)):
            lat = lats[j]
            wt = wtlist[j]
            t = tlist[j]      
            tave += t * wt * renorm
        
        return tave










def tempWtsDiskAvg(lats,cx,cy,beamx,beamy,theta,Titan_radius,Titan_dist,top_atm,dx,subobslat,ccw,diagnostics):
    """ Takes in definition of elliptical beam at arbitrary location on Titan's disk.  Returns weightings to be used in .spx header.
    cx,cy are grid of coords of center of beam in km from center of Titan in km. """

    Titan_dist = Titan_dist*149597870 #to km
    a = Titan_dist*np.radians(np.arcsin(beamx/3600))
    b = Titan_dist*np.radians(np.arcsin(beamy/3600))

    subobslat = np.deg2rad(subobslat)
    ccw = np.deg2rad(ccw)

    #Finding vector of true north of Ttan
    northx = -np.sin(ccw)*np.cos(subobslat)
    northy = np.cos(ccw)*np.cos(subobslat)
    northz = np.sin(subobslat) 
    
    #create grid z where each point represents Titan latitude
    lim = Titan_radius + top_atm + a #put a here so map extends out to a full beam past top of atmosphere
    xx = slice(-lim,lim+dx,dx)
    y,x = np.mgrid[xx,xx]
    z = y*x*0.0 #initializes array z, which becomes the latitudes on the 2-d Titan map - confusing I know
    with np.errstate(divide='ignore',invalid='ignore'): #We actually want all y > Titan_radius + top_atm to be nans, so the invalid inputs to arcsin are helping here
        zcoord = np.sqrt((Titan_radius+top_atm)**2 - x**2 - y**2) #these are the actual z-coordinates (distance from Titan center to observer) at each x,y point
        dprod = (northx*x + northy*y + northz*zcoord)/(Titan_radius+top_atm) #dot product of north pole vector and each vector in model planet
        z = 90 - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid
        for j in range(1,len(lats)):
            z[np.logical_and(z >= lats[j-1], z < lats[j])] = lats[j-1] #discretize z into the latitudes given in the input data

    #make grid rad where each point represents distance from center of Titan in km
    rad = np.sqrt(x**2 + y**2)

    #Handle bad and out-of-bounds values
    z[z == -1.0] = float('NaN')
    z[rad >= Titan_radius + top_atm] = float('NaN')

    ones = np.copy(z) * 0.0 + 1 #just a grid of ones
    #compute normalized weights
    wts = {}
    for val in lats:
        garr = ones[np.where(z == val)]
        wts[val] = sum(garr)
    # gnanarr =  g[np.where(np.isnan(z))] #treating NaN values such that weighting will not add up to one
    #s = sum(wts.values())+sum(gnanarr)
    s = sum(wts.values())
    for key,val in wts.items():
        val = float(val)/float(s)
        wts[key] = val

    meanlat = sum([val*key for key,val in wts.items()])
        
    ### PLOTS ###    
    #latitude map
    if diagnostics and k == 3:
        fig1 = plt.figure(figsize = (15,15))
        ax = fig1.add_subplot(111)
        ax.imshow(z, cmap='RdBu',origin='lower')
        plt.show()
        
    #Gaussian laid onto circular region
    if diagnostics and k == 3:
        fig3 = plt.figure(figsize = (15,15))
        ax = fig3.add_subplot(111)
        ax.imshow(np.multiply(g,z*0.0 + 1),cmap = 'RdBu',origin='lower')
        plt.show()
        
    ## #Gaussian convolved with latitude map
    ## if diagnostics and k == 3:
    ##     conv = np.multiply(g,z)
    ##     fig2 = plt.figure(figsize = (15,15))
    ##     ax = fig2.add_subplot(111)
    ##     ax.imshow(conv, cmap='RdBu',origin='lower')
    ##     plt.show()
        
    return wts, meanlat
    



######################################################
####################### CODE #########################
######################################################

if usecasa:
    [beamx,beamy,theta,refx,refy,pixszx,pixszy] = casa_extract(img)
theta = np.radians(theta+90.0) #change to what's required by computeWeights

if diskavg:
    cx = [0.0]
    cy = [0.0]

(temps,lats) = tmap_extract(infile,standard_h)
 
#average temperatures at each latitude, turn profile into output file
fout = open(outfile,'w')
fout.write('### Altitude (km)   -   Temperature (K)   -   Mean Latitude at this Altitude ###\n')
profiles = {}
for valx in cx:
    profiles[valx] = {}
    for valy in cy:
        if valy != cy[0]:
            print('-------------------------------------')
        print('(cx,cy) = ('+str(valx)+','+str(valy)+')')
        fout.write('# Beam centered at x, y = '+str(valx)+', '+str(valy)+' km from center of planet #\n')
        tprof = []
        hprof = []    
        for k in range(len(standard_h)):
            ts = temps[k]
            if diskavg:
                wts,meanlat = tempWtsDiskAvg(lats,valx,valy,beamx,beamy,theta,Titan_radius,Titan_dist,standard_h[k],dx,subobslat,ccw,diagnostics)
            else:
                wts,meanlat = tempWts(lats,valx,valy,beamx,beamy,theta,Titan_radius,Titan_dist,standard_h[k],dx,subobslat,ccw,diagnostics)
            tave = meanTemp(wts,lats,ts)
            if tave <= 0.0 or np.isnan(tave):
                print('INFO: Zero data points at altitude '+str(standard_h[k])+' km')
            else:
                tprof.append(tave)
                hprof.append(standard_h[k])
                fout.write(str(standard_h[k])+'   '+str(tave)+'   '+str(meanlat)+'\n')
        profiles[valx][valy] = [tprof,hprof]

if diskavg:
    fig = plt.figure()
    ax = fig.add_axes([0.08,0.15,0.9,0.8])
    disk, = ax.plot(profiles[cx[0]][cy[0]][0],profiles[cx[0]][cy[0]][1],color='r')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    plt.legend([disk],['Disk Average'],loc='upper left')

    plt.show()
    fig.savefig(outfigure,bbox='None')

'''
else:
    fig = plt.figure()
    ax = fig.add_axes([0.08,0.15,0.9,0.8])
    south, = ax.plot(profiles[cx[0]][cy[0]][0],profiles[cx[0]][cy[0]][1],color='r')
    equator, = ax.plot(profiles[cx[0]][cy[1]][0],profiles[cx[0]][cy[1]][1],color='g')
    north, = ax.plot(profiles[cx[0]][cy[2]][0],profiles[cx[0]][cy[2]][1],color='b')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    plt.legend((south,equator,north),('South Pole','Equator','North Pole'),loc='upper left')

    plt.show()
    fig.savefig(outfigure,bbox='None')
'''