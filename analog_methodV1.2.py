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
import time

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

#Iterieren über ano_roll
#Für jeden Tag die Pseudo PCs berechnen
#Für jeden Tag die pcs aus dem solver (index des doy) berechnen
#Time dimension für pcs korrigieren ([ len(Time) / no_of_years ] zum 1.Tag dazufügen,...
#... dayofyear abgreifen und ausdehen auf alle Jahre)
#Jahr das gerade betrachtet wird löschen
#Datum der pcs richtig zuordnen
#Norm und Minimimum berechnen

#%%
time_start = data_ano.time.to_series().dt.strftime('%Y-%m-%d')[0]
time_end = data_ano.time.to_series().dt.strftime('%Y-%m-%d')[-1]
timeseries = pd.date_range(time_start, time_end, freq='D')
#print(timeseries)
analoga = []
#%%


t_start = time.time()
for day in timeseries:
    query_start = day - datetime.timedelta(days=10)
    query_end = day + datetime.timedelta(days=10)
    query_period = pd.date_range(query_start, query_end, freq='D')
    query_doy = query_period.dayofyear
    query_period = query_period[query_period >= data_ano.time.to_series()[0]]
    query_period = query_period[query_period <= data_ano.time.to_series()[-1]]

    ds = data_ano.sel(time = data_ano.time.dt.dayofyear.isin(query_doy))
    ds = ds.sel(time = ~ds.time.dt.year.isin(day.year))

    doy = data_ano.sel(time=day).dayofyear.values
    ds_day = data_ano.sel(time=day)

    ds.coords['time'].attrs['axis'] = 'T'
#    t_end_tmp = time.time()
#    print("Preproc: ", t_end_tmp - t_start)
    
    ds_iris_r = ds.r.to_iris()
    ds_iris_q = ds.q.to_iris()
    ds_iris_msl = ds.msl.to_iris()
    cube_list = iris.cube.CubeList([ds_iris_r, ds_iris_q, ds_iris_msl])
    ds_iris = cube_list.merge()
#    t_end_tmp = time.time()
#    print("iris_conversion: ", t_end_tmp - t_start) 
    
    solver = MultivariateEof(ds_iris, weights=None)
#    t_end_tmp = time.time()
#    print("solver: ", t_end_tmp - t_start)
    
    n = 1
    eof_var = solver.varianceFraction(neigs=n)
    while np.sum(eof_var.data) <= 0.9:
        n += 1
        eof_var = solver.varianceFraction(neigs=n)
    
    pc = solver.pcs(npcs=n, pcscaling=1)
    da_pc = xr.DataArray.from_iris(pc).rename('pc')
#    t_end_tmp = time.time()
#    print("pc: ", t_end_tmp - t_start)
    
    ds_iris_r_query_day = ds_day.r.to_iris()
    ds_iris_q_query_day = ds_day.q.to_iris()
    ds_iris_msl_query_day = ds_day.msl.to_iris()
    pseudo_pc = solver.projectField([ds_iris_r_query_day, ds_iris_q_query_day,ds_iris_msl_query_day], neofs=n)
    da_pseudo_pc = xr.DataArray.from_iris(pseudo_pc).rename('pseudo_pc')
#    t_end_tmp = time.time()
#    print("ppc: ", t_end_tmp - t_start)
    
    norm = (( da_pc - da_pseudo_pc )**2).sum(dim='pc')
    minimum_idx = np.argmin(norm).values
    analogon = ds.time.to_series()[minimum_idx]
    analoga.append(analogon)
    print('Found analogon for: ', day)
    print('matching day (analogon): ', analogon)
    
    t_end_tmp = time.time()
    print("total time: ", t_end_tmp - t_start)
    



#%%

