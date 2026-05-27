import scipy.io as sio
from scipy.interpolate import griddata
import numpy as np
import pygmt
import xarray
import datetime as dt
import os
import imageio

solar_act = 'high'
filename = 'NmF2_pred_'+solar_act+'daily.mat'

year = 2012
month = 12
dom = 17
UT = 14
minute = 0
date_dur = 2  # hours
date_res = 120  # minutes
lat_res = 2.5
lon_res = 5
Lat_res = 0.25*5
Lon_res = 0.5*5

names = ['data']  # 'GP', 'data', 'IRI',

# data loading
data = sio.loadmat(filename)
NmF2_data = data['NmF2_data']

# interpolation
lat_axis = np.arange(-90, 90+lat_res, lat_res)
lon_axis = np.arange(-180, 180+lon_res, lon_res)
Lat_axis = np.arange(-90, 90+Lat_res, Lat_res)
Lon_axis = np.arange(-180, 180+Lon_res, Lon_res)

coords = np.zeros([len(lon_axis)*len(lon_axis), 2])
grid_lat, grid_lon = np.meshgrid(lon_axis, lat_axis)
coords[:, 0] = grid_lat.reshape(-1, 1).squeeze()
coords[:, 1] = grid_lon.reshape(-1, 1).squeeze()

grid_lat, grid_lon = np.meshgrid(Lon_axis, Lat_axis)
date_sta = dt.datetime(year, month, dom, UT, minute)

# draw figures

fig = pygmt.Figure()

for n in range(NmF2_data.shape[0]):

    date = date_sta+dt.timedelta(minutes=date_res*n)
    date_folder = 'Cases/'+date.strftime("%m-%d-%Y-%H:%M:%S")
    if os.path.isdir(date_folder):
        pass
    else:
        os.mkdir(date_folder)

    for i in range(len(names)):

        X_t = data['NmF2_'+names[i]]
        X_t[n, np.isnan(X_t[n])] = np.nanmean(X_t[n])
        X = np.zeros(X_t[n].shape)
        X = X_t[n]
        X = X.reshape(-1, 1).squeeze()
        X_grid = griddata(coords, X, (grid_lat, grid_lon), method='cubic')
        X = xarray.DataArray(data=X_grid,
                             coords={'lat': Lat_axis, 'lon': Lon_axis},
                             dims=['lat', 'lon'])

        print('i=', i)
        fig.basemap(region="g",
                    # projection="W6i",
                    frame=True,
                    # V='d',
                    projection="Cyl_stere/30/-20/8i")

        fig.grdimage(X, cmap='jet')
        fig.coast(shorelines="0.5p,black")

        fig.colorbar(position="jBC+h@;white",
                     frame=["xaf",
                            "y+l10@+5@+el/cm@+3"])
        filename = date_folder+'/'+names[i]+'.png'
        fig.savefig(filename,
                    show=True, dpi=300)
        fig.close()

'''
# create .gif
for i in range(len(names)):
    gifname = 'Cases/gif/'+names[i]+'_' +\
        date_sta.strftime("%m-%d-%Y-%H:%M:%S") + '_' +\
        date.strftime("%m-%d-%Y-%H:%M:%S")+'.mp4'

    with imageio.get_writer(gifname, mode='I') as f:
        for n in range(NmF2_data.shape[0]):
            date = date_sta+dt.timedelta(minutes=date_res*n)
            filename = 'Cases/'+date.strftime("%m-%d-%Y-%H:%M:%S") +\
                '/'+names[i]+'.png'
            image = imageio.imread(filename)
            f.append_data(image)
'''
