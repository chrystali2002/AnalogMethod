import xarray as xr
import numpy as np
import dask
#from reanalysis import GCM_data
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
import cloudpickle

#path = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/'AnalogMethod/
path = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/'

#%%
#data = GCM_data(path)
#data.to_dailymean()
#data.climatology()
#data_ano = data.anomaly()

#%%
# Load anomaly data.
data_ano = xr.open_mfdataset(path+'anom_all_1979-2018.nc', chunks={'time': 5000})

#%%
# Create time series for which the analoga should be calculated.
time_start = '1970-01-01'
time_end = '1970-12-31'

#time_end = data_ano.time.to_series().dt.strftime('%Y-%m-%d')[-1]
timeseries = pd.date_range(time_start, time_end, freq='D')

# Define number of analogons to be found for each day in timeseries.
no_analoga = 5

# Define number of years in dataset.
no_years = len(data_ano.groupby('time.year'))

# Create empty array to be filled with 1st,2nd,3rd,... analoga.
# First column represents days for which analoga should be found.
# Second column = 1st analogon.
# Third column = 2nd analogon. etc.
analoga = np.zeros((len(data_ano.time), no_analoga+1), dtype='datetime64[s]')
analoga[:,0] = data_ano.time

#%%
# Run time of script is observed with t_start and t_end respectively.
t_start = time.time()

# Iterate over earlier specified time series.
for i, day in enumerate(timeseries):
    
    # Define query window for which an analogon should be found:
    # 10 days before and 10 days after "day" in each year
    query_start = day - datetime.timedelta(days=10)
    query_end = day + datetime.timedelta(days=10)
    query_period = pd.date_range(query_start, query_end, freq='D')
    
    # Select all doy of query_period.
    query_doy = query_period.dayofyear

    # Define dataset to work with.
    # Select all doy in dataset to use for pc (and eof) calculation.
    ds = data_ano.sel(time = data_ano.time.dt.dayofyear.isin(query_doy))

    # Define dataset for current doy (required for ppc calculation)
    ds_doy = data_ano.sel(time = data_ano.time.dt.dayofyear.isin(day.dayofyear))

    # Assign attrs 'T' to time dimension (required for EOF analysis package)
    ds.coords['time'].attrs['axis'] = 'T'
    
    # Convert DataArrays to iris cubes.
    ds_iris_r = ds.r.to_iris()
    ds_iris_q = ds.q.to_iris()
    ds_iris_msl = ds.msl.to_iris()
    
    # Merge iris cubes to list of iris cubes.
    cube_list = iris.cube.CubeList([ds_iris_r, ds_iris_q, ds_iris_msl])
    ds_iris = cube_list.merge() 
    
    # Conduct multivariate EOF analysis.
    solver = MultivariateEof(ds_iris, weights=None)
    
    # Calculate pcs (Principal Components) until the explained variance
    # exceeds 0.9
    n = 1
    eof_var = solver.varianceFraction(neigs=n)
    while np.sum(eof_var.data) <= 0.9:
        n += 1
        eof_var = solver.varianceFraction(neigs=n)
    
    pc = solver.pcs(npcs=n, pcscaling=1)
    
    # Transform iris cube of pcs to DataArray (xarray)
    da_pc = xr.DataArray.from_iris(pc).rename('pc')
    
    # Convert DataArrays of current doy to iris cubes.
    # ppc calculation is vectorized and therefore carried out simultaneously
    # for each doy.
    ds_iris_r_query_doy = ds_doy.r.to_iris()
    ds_iris_q_query_doy = ds_doy.q.to_iris()
    ds_iris_msl_query_doy = ds_doy.msl.to_iris()

    # Merge iris cubes to list of iris cubes.
    cube_list_query_doy = iris.cube.CubeList([ds_iris_r_query_doy, \
                                    ds_iris_q_query_doy, ds_iris_msl_query_doy])
    ds_iris_query_doy = cube_list_query_doy.merge()     

    # Calculate ppcs (Pseudo-PCs)
    pseudo_pc = solver.projectField(ds_iris_query_doy, neofs=n)

    # Transform iris cube of pseudo-pcs to DataArray (xarray)
    da_pseudo_pc = xr.DataArray.from_iris(pseudo_pc).rename('pseudo_pc')

    for y in range(0, no_years):
        # Caclulate norm.
        norm = (( da_pc - da_pseudo_pc[y] )**2).sum(dim='pc')
        # Delete current year y from norm array.
        norm = norm.sel(time = ~norm.time.dt.year.isin(da_pseudo_pc.time.to_series()[y].year))
        str_curr_day = da_pseudo_pc.time.to_series().dt.strftime('%Y-%m-%d')[y]

        print('Looking for analoga for: ', str_curr_day)
        for a in range(1,no_analoga+1):
            # Find analogon (day in dataset) through localization of minimum norm.
            minimum_idx = np.argmin(norm).values
            # Temporarily delete current year y from ds.
            ds_tmp = ds.sel(time = ~ds.time.dt.year.isin(da_pseudo_pc.time.to_series()[y].year))
            analogon = ds_tmp.time.to_series()[minimum_idx]
            
            # Add analogon to analogon array
            # row... select row of current day in analoga array.
            row = i + y*no_years
            analoga[row][a] = analogon
            
            # Set minimum value to nan in norm array to find 2nd, 3rd, ... minimum.
            norm = norm.where(norm.values != norm[minimum_idx].values)
    
            print(str(a) +  '.analogon: ', str(analogon)[:10])
    
    t_end_tmp = time.time()
    print("total time: ", str(datetime.timedelta(seconds=t_end_tmp - t_start)), '\n')

# Save analogon list as pickle file.
print('Analoga : \n', analoga)
print("Saving analoga array as 'analoga_list.p'")
cloudpickle.dump( analoga, open( "analoga_list_" + time_start + "_" + \
                                    time_end + ".p", "wb" ) )
