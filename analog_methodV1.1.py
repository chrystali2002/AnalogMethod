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
#print(data.ano)
#ano_roll = data.ano.rolling(time=21, center=True).construct('window_dim')
ano_roll = data_ano.rolling(time=21, center=True).construct('window_dim')

n = 3 # number of eofs, pc, etc.
first = 1
eof_datasets = []
pc_dataarrays = []

query_day = datetime.datetime(1980,3,6)
query_start = query_day - datetime.timedelta(days=10)
query_end = query_day + datetime.timedelta(days=10)
query_period = pd.date_range(query_start, query_end, freq='D')

#print(query_period.dayofyear)

#print(query_period.day)
#print(ano_roll.time)
#print(query_period.month.values)
#print(ano_roll.time.dt.month.values)
#print(ano_roll.sel(time.dt.month=query_period.month))
#print(ano_roll.where(ano_roll.time.dt.dayofyear == query_period.dayofyear))
ds_ano_roll_query = ano_roll.sel(time = (ano_roll.time.dt.dayofyear.isin(query_period.dayofyear)))
print(ds_ano_roll_query.time)
print(ano_roll.dayofyear.values)
#print(ano_roll.time.resample('D'))

#ano_roll_month_sel = ano_roll.sel(time = (ano_roll.time.dt.month.isin(query_period.month.values)))
#print(ano_roll_month_sel)
#ano_roll_query_period = ano_roll_month_sel.sel(time = ano_roll_month_sel.time.dt.day.isin(query_period.day.values))
#print(ano_roll_query_period)
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
    n = 1
    eof_var = solver.varianceFraction(neigs=n)
    while np.sum(eof_var.data) <= 0.9:
        n += 1
        eof_var = solver.varianceFraction(neigs=n)

    pc = solver.pcs(npcs=n, pcscaling=1)
    da_pc = xr.DataArray.from_iris(pc).rename('pc')
    print(da_pc)

#%%


#ds_ano_roll_query = ano_roll.sel(time=query_period)
#print(ds_ano_roll_query)
t_start = time.time()

for _,ds in ds_ano_roll_query.groupby('time.dayofyear'):
#for _,ds in ano_roll.groupby('time'):


        print(ds)
        ds = ds.rename({'time':'time_old'})
        print(ds)
        # Set doy as string (also start point for time series for iris conversion)
        #time_start = ds.time_old.to_series().dt.strftime('%Y-%m-%d')#[0]
        #time_start = query_day

        # Stack times and window dimension (years * window_dim)
        #ds = ds.stack(time=['time_old', 'window_dim']).transpose('time','latitude','longitude').dropna('time')
        ds = ds.stack(time=['time_old', 'window_dim']).transpose('time','lat','lon').dropna('time')

        # Assign attr 'T' to time coordinate (required for iris conversion)
        ds.coords['time'].attrs['axis'] = 'T'

        # Create monotonous time series (required for iris conversion).
        # Start of time series is corresponding doy.
        #time_series = pd.DatetimeIndex(start=time_start,periods=ds['time'].shape[0],freq='D')
        #print(len(time_series))
        # Set new time coordinate
        #print(ds_ano_roll_query.time)
        
        ds = ds.assign_coords(time=ds_ano_roll_query.time)
        #ds = ds.assign_coords(time=time_series)
        t_end_tmp = time.time()
        print(time_start[-5:])
        print("Total Duration: ", t_end_tmp - t_start)
        # Convert DataArrays of individual variables to iris cubes
        # (required for multivariate EOF)        
        ds_iris_r = ds.r.to_iris()
        ds_iris_q = ds.q.to_iris()
        cube_list = iris.cube.CubeList([ds_iris_r, ds_iris_q])
        ds_iris = cube_list.merge()
        
        # Calculate multivariate EOF
        solver = MultivariateEof(ds_iris, weights=None)
        #eof = solver.eofs(neofs=n)
        pc = solver.pcs(npcs=n, pcscaling=1)
        #eigval = solver.eigenvalues(neigs=n)
        #eof_var = solver.varianceFraction(neigs=n)
        #print(pc)
        # Convert to xarray
        da_pc_tmp = xr.DataArray.from_iris(pc).rename('pc')
        print(len(da_pc_tmp.time))
        #print(da_pc_tmp['time'])
        #print(time_series)
        #da_pc_tmp['time'] = time_series[0]
        pc_dataarrays.append(da_pc_tmp)

        #print(ds_pc)
        #print(ds_ppc)
        #print(da_pc_tmp)

        #if first == 1:
            # Calculate Pseudo PC
        ds_iris_r_query_day = ds_iris_r.extract(iris.Constraint(time=query_day))
        ds_iris_q_query_day = ds_iris_q.extract(iris.Constraint(time=query_day)) 
        #print(ds_iris_r_query_day)
        pseudo_pc = solver.projectField([ds_iris_r_query_day, ds_iris_q_query_day], neofs=n)
        #print(pseudo_pc)
        da_ppc = xr.DataArray.from_iris(pseudo_pc).rename('ppc')
        #first += 1
        
        print(da_pc_tmp)
        print(da_ppc)
        # Convert to xarray
#        da_eof_r = xr.DataArray.from_iris(eof[0]).rename('eof_r')
#        da_eof_q = xr.DataArray.from_iris(eof[1]).rename('eof_q')
#        ds_eof_tmp = xr.merge([da_eof_r, da_eof_q])
#        
#        ds_eof_tmp['time'] = time_series[0]
#        
#        eof_datasets.append(ds_eof_tmp)

        #da_eof = xr.concat(da_eof_r, dim='time')
        #print(ds_eof_tmp)
#        first +=1
#        if first == 4:
#            break

da_pc = xr.concat(pc_dataarrays, dim='time')
#%%
print(da_pc)
print(da_ppc)
norm = (( da_pc - da_ppc )**2).sum(dim='pc')
minimum_idx = np.argmin(norm).values
print(norm)
print(minimum_idx)  
#ds_eof = xr.concat(eof_datasets, dim='time')
#print(ds_eof)
print(ds_ano_roll_query)
#%%
ds_eof['eof_r'].isel(eof=0, time=2).plot()

#plt.show()
