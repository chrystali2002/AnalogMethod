import xarray as xr
import numpy as np
import dask

"""
reanalysis.py creates class from era5 netcdf datasets imported with xarray

Input:
------
    - xarray dataset

Output:
-------
    - python class which can:

        + calculate DAILY MEAN ( if you only have hourly data)

        + calculate daily CLIMATOLOGY with a 21 day window (+- 10 days from actual day)

        + calculate daily NORMALIZED ANOMALIES

path with data FOR TESTING:
from reanalysis import GCM_data
GCM_data('/Users/kristoferhasel/Desktop/UNI/MASTER/Klimamodelle/UE/AnalogMethod/')

"""

class GCM_data:
    def __init__(self, datapath):
        self.datapath = datapath
        self.ds = xr.open_mfdataset(self.datapath+'*.nc', 
                        chunks={'time': 504,'latitude': 141, 'longitude': 141})
        self.vars = list(self.ds.data_vars)


    def to_dailymean(self):
        print('Start: Calculating daily mean')
        # calculates daily mean and creates bigger chunksize for rolling window in the steps ahead
        self.dmean = self.ds.resample(time='1D').mean().chunk(
                        {'time': 508,'longitude': 141, 'latitude':141})
        print('Finished: Calculating daily mean')
        return self.dmean


    def climatology(self):
        try:
            self.dmean
        except AttributeError:
            self.dmean = self.to_dailymean()

        # calculates the climatology and it's standard derivation for further anomaly calculation
        # climatology is calculated for dayofyear of Target Day ±10days (pool)
        # standard derivation is calculated for dayof year of the pool

        print('Start: Calculating climatology')
        # Construct window dimension (centered).
        dmean_roll = self.dmean.rolling(time=21, center=True).construct('window_dim')       
        # Calculate climatology for each day of year (doy), based on time window.
        self.clim = dmean_roll.groupby('time.dayofyear').mean(dim=['window_dim','time'])       
        # Calculate standard deviation for each day of year (doy) for all dimensions.
        self.ss = dmean_roll.groupby('time.dayofyear').std(dim=xr.ALL_DIMS)
        
        print('Finished: Calculating climatology')
        return self.clim, self.ss


    def anomaly(self):
        try:
            self.clim
        except AttributeError:
            self.clim, self.ss = self.climatology()
            
        print('Start: Calculating anomalies')
        # calculates daily normalized anomalies from TD + pool
        self.ano = self.dmean.groupby('time.dayofyear') - self.clim
        self.ano = self.dmean.groupby('time.dayofyear') / self.ss
        
        self.ano = self.ano.chunk({'time': -1})
        
        print('Finished: Calculating anomalies')
        return self.ano
