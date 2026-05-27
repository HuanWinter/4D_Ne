import scipy.io as sio
from scipy.interpolate import griddata
import numpy as np
import pygmt
import xarray

solar_act = 'high'
filename = 'NmF2_pred_'+solar_act+'.mat'

year = 2013
month = 3
dom = 17
UT = 8
minute = 30
date_dur = 24  # hours
date_res = 30  # minutes
lat_res = 5
lon_res = 10
Lat_res = 0.5
Lon_res = 1

name = ['GP', 'data', 'IRI']

# data loading
data = sio.loadmat(filename)
NmF2_GP = data['NmF2_GP']
NmF2_NN = data['NmF2_NN']
NmF2_IRI = data['NmF2_IRI']
NmF2_data = data['NmF2_data']
NmF2_std = data['NmF2_std']

n = 0

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

# draw figures

fig = pygmt.Figure()
fig.subplot(directive="begin", row=len(name), col=1, dimensions="s3c/3c")

print(len(name))

for i in range(len(name)):

    X_t = data['NmF2_'+name[i]]
    X = np.zeros(X_t[n].shape)
    X = X_t[n]/10e5
    X = X.reshape(-1, 1).squeeze()
    X_grid = griddata(coords, X, (grid_lat, grid_lon), method='cubic')
    X = xarray.DataArray(data=X_grid,
                         coords={'lat': Lat_axis, 'lon': Lon_axis},
                         dims=['lat', 'lon'])

    print('i=', i)
    fig.subplot(directive="set", row=i, col=n)
    fig.basemap(region="g",
                # projection="W6i",
                frame=True,
                # V='d',
                projection="Cyl_stere/30/-20/8i")

    fig.grdimage(X, cmap='jet')
    fig.coast(shorelines="0.5p,black")

    # fig.colorbar(position="jBC+h@;white",
                 # frame=["xaf",
                        # "y+l10@+5@+el/m@+3"])

    
fig.subplot(directive="end")
fig.savefig('NmF2_all.png',
            show=True, dpi=300)
