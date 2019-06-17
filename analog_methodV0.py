import xarray as xr
import numpy as np
import dask
from reanalysis import GCM_data
#from eofs.iris import MultivariateEof
from eofs.iris import Eof
#from eofs.xarray import Eof
from eofs.multivariate.iris import MultivariateEof
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import pandas as pd
import datetime

path = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/AnalogMethod/'

#%%
data = GCM_data(path)
data.to_dailymean()
data.climatology()
data.anomaly()

#%%
#print(data.ano)
ano_roll = data.ano.rolling(time=21, center=True).construct('window_dim')
n = 5 # number of eofs, pc, etc.
first = 1
eof_datasets = []

for _,ds in ano_roll.groupby('time.dayofyear'):
        ds = ds.rename({'time':'time_old'})
        
        # Set doy as string (also start point for time series for iris conversion)
        time_start = ds.time_old.to_series().dt.strftime('%Y-%m-%d')[0]

        # Stack times and window dimension (years * window_dim)
        ds = ds.stack(time=['time_old', 'window_dim']).transpose('time','latitude','longitude').dropna('time')

        # Assign attr 'T' to time coordinate (required for iris conversion)
        ds.coords['time'].attrs['axis'] = 'T'

        # Create monotonous time series (required for iris conversion).
        # Start of time series is corresponding doy.
        time_series = pd.DatetimeIndex(start=time_start,periods=ds['time'].shape[0],freq='D')

        # Set new time coordinate
        ds = ds.assign_coords(time=time_series)


        # Convert DataArrays of individual variables to iris cubes
        # (required for multivariate EOF)        
        ds_iris_r = ds.r.to_iris()
        ds_iris_q = ds.q.to_iris()
        cube_list = iris.cube.CubeList([ds_iris_r, ds_iris_q])
        ds_iris = cube_list.merge()
        
        # Calculate multivariate EOF
        solver = MultivariateEof([ds_iris_r, ds_iris_q], weights=None)
        eof = solver.eofs(neofs=n)
        pc = solver.pcs(npcs=n, pcscaling=1)
        eigval = solver.eigenvalues(neigs=n)
        eof_var = solver.varianceFraction(neigs=n)
        pseudo_pc = solver.projectField(ds_iris, neofs=n)
        
        #print(eof)
        
        # Convert to xarray
        da_eof_r = xr.DataArray.from_iris(eof[0]).rename('eof_r')
        da_eof_q = xr.DataArray.from_iris(eof[1]).rename('eof_q')
        ds_eof_tmp = xr.merge([da_eof_r, da_eof_q])
        
        ds_eof_tmp['time'] = time_series[0]
        
        eof_datasets.append(ds_eof_tmp)

        #da_eof = xr.concat(da_eof_r, dim='time')
        print(ds_eof_tmp)
        first +=1
        if first == 4:
            break
        
ds_eof = xr.concat(eof_datasets, dim='time')
print(ds_eof)

#%%
ds_eof['eof_r'].isel(eof=0, time=2).plot()

#plt.show()
