import xarray as xr
import numpy as np
import dask
from reanalysis import GCM_data
from eofs.iris import Eof
from eofs.multivariate.iris import MultivariateEof
import iris
import pandas as pd
import datetime
import time
import cloudpickle

#path = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/'AnalogMethod/
path = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/'

#%%
#data = GCM_data(path)
#data.to_dailymean()
#data.climatology()
#data.anomaly()

#%%
# Load anomaly data.
data_ano = xr.open_mfdataset(path+'anom_all_1979-2018.nc', chunks={'time': 5000})

#%%
#ano_roll = data.ano.rolling(time=21, center=True).construct('window_dim')
ano_roll = data_ano.rolling(time=21, center=True).construct('window_dim')

solver_list = []
pc_list = []

#%%
t_start = time.time()
first = 1
for _,ds in ano_roll.groupby('time.dayofyear'):
    ds = ds.rename({'time':'time_old'})
    print(ds)
    
    # Set doy as string (also start point for time series for iris conversion)
    time_start = ds.time_old.to_series().dt.strftime('%Y-%m-%d')[0]

    # Stack times and window dimension (years * window_dim)
    #ds = ds.stack(time=['time_old', 'window_dim']).transpose('time','latitude','longitude').dropna('time')
    ds = ds.stack(time=['time_old', 'window_dim']).transpose('time','lat','lon').dropna('time')

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
    solver = MultivariateEof(ds_iris, weights=None)
    solver_list.append(solver)
    
    t_end_tmp = time.time()
    print(time_start[-5:])
    print("Total Duration: ", t_end_tmp - t_start)
    
    first += 1
    if first == 6:
        break
        
#%%
cloudpickle.dump( solver_list, open( "solver_list.p", "wb" ) )
#test = cloudpickle.load( open( "solver_list.p", "rb" ) )