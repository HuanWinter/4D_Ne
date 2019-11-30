from netCDF4 import Dataset
import numpy as np
import progressbar
import datetime as dt
import matplotlib.pyplot as plt
import ipdb


################################### Global coefficients #################

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim)
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

def ncread(nc_fid):

    lats = nc_fid.variables['lat'][:]  # extract/copy the data
    lons = nc_fid.variables['lon'][:]
    time = nc_fid.variables['time'][:]
    ut = nc_fid.variables['ut'][:]
    lev = nc_fid.variables['lev'][:]
    Ti = nc_fid.variables['TI'][:]
    Te = nc_fid.variables['TE'][:]
    Ne = nc_fid.variables['NE'][:]
    Z = nc_fid.variables['ZGMID'][:]

    MI = np.zeros(Ne.shape)

    for a in range(Ne.shape[0]):
        for b in range(Ne.shape[1]):
            for d in range(Ne.shape[3]):
                ind_max = np.argmax(Ne[a,b,:,d])
                for c in range(ind_max, Ne.shape[2]): #nd_max,
                    MI[a,b,c,d] = 1

    Lon, Lat, Lev, UT = np.meshgrid(lons, lats, lev, ut)

    Lon = np.reshape(Lon, (1,-1)).squeeze()
    Lat = np.reshape(Lat, (1,-1)).squeeze()
    Lev = np.reshape(Lev, (1,-1)).squeeze()
    UT = np.reshape(UT, (1,-1)).squeeze()
    Te = np.reshape(Te, (1,-1)).squeeze()
    Ti = np.reshape(Ti, (1,-1)).squeeze()
    Z = np.reshape(Z, (1,-1)).squeeze()/1e5
    MI = np.reshape(MI, (1,-1)).squeeze()

    LT = UT + Lon/15

    LT[np.where(LT>=24)] = LT[np.where(LT>=24)]-24
    LT[np.where(LT<0)] = LT[np.where(LT<0)]+24

    return Lon, Lat, Z, UT, Te, MI

def TIE_grid_h_time(Lon, Lat, Alt, LT, y, MI, \
    time_rang, h_rang, time_res, h_res, coords, coords_buffer, y_rang):

    Y = np.zeros([int(np.diff(h_rang)/h_res)+1, int(np.diff(time_rang)/time_res)+1])
    Num = np.zeros([int(np.diff(h_rang)/h_res)+1, int(np.diff(time_rang)/time_res)+1])

    for m in range(time_rang[0],time_rang[1],time_res):
        for n in range(h_rang[0],h_rang[1],h_res):
            '''
            index = np.where((LT>=m) & (LT<m+time_res) &\
                             (Alt>=n) & (Alt<n+h_res) &\
                             (Lat>=coords[0]-coords_buffer[0]) &\
                             (Lat< coords[0]+coords_buffer[0]) &\
                             (Lon>=coords[1]-coords_buffer[1]) &\
                             (Lon< coords[1]+coords_buffer[1]) &\
                             (y>=y_rang[0]) & (y<y_rang[1]) &\
                             (MI==1))
            '''
            index = np.where((LT>=m) & (LT<m+time_res) &\
                             (Alt>=n) & (Alt<n+h_res) &\
                             (Lat>=coords[0]-15) &\
                             (Lat< coords[0]+15) &\
                             (Lon>=coords[1]-180) &\
                             (Lon< coords[1]+180) &\
                             (y>=y_rang[0]) & (y<y_rang[1]) &\
                             (MI==1))
	    
            #ipdb.set_trace()
            if index[0].shape[0] == 0:
                Y[(n-h_rang[0])//h_res, (m-time_rang[0])//time_res] = np.nan
                #print(index)
                continue

            Y[(n-h_rang[0])//h_res, (m-time_rang[0])//time_res] = np.mean(y[index[0]])
            Num[(n-h_rang[0])//h_res, (m-time_rang[0])//time_res] \
                    = index[0].shape[0]

    Y[:,0] = Y[:,-1]
    Num[:,0] = Num[:,-1]

    return Y, Num
