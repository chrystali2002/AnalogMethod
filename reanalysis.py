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

        + calculate CLIMATOLOGY with a 21 day window (+- 10 days from actual day)

        + calculate NORMALIZED ANOMALIES

path with data FOR TESTING:
GCM_data('/Users/kristoferhasel/Desktop/UNI/MASTER/Klimamodelle/UE/AnalogMethod/')

"""

class GCM_data:
    def __init__(self, datapath):
        self.datapath = datapath
        self.ds = xr.open_mfdataset(self.datapath+'*.nc', chunks={'time': 504,
                                                                  'latitude': 141, 'longitude': 141})
        self.vars = list(self.ds.data_vars)
        for i in self.vars:
            self.ds[i]

    def to_dailymean(self):
        self.dmean = {}
        for i, val in enumerate(self.vars):
            self.dmean[val] = self.ds[val].resample(time='1D').mean(axis=0).chunk({'time': 508,
                                                                          'longitude': 141, 'latitude':141})
        return self.dmean

    def climatology(self):
        try
            self.dmean
        except NameError:
            self.dmean = self.to_dailymean()
        self.clim = {}
        self.ss = {}
        for i, val in enumerate(self.vars):
            self.clim[val] = self.dmean[val].rolling(time=21, center=True).mean().dropna('time').groupby('time.dayofyear').mean(axis=0)
            ss_tmp = self.dmean[val].rolling(time=21, center=True).std().dropna('time')
            self.ss[val] = ss_tmp.std(dim={'longitude','latitude'}).groupby('time.dayofyear').std(axis=0)
        return self.clim, self.ss

    def anomaly(self):
        try:
            self.clim
        except NameError:
            self.clim, self.ss = self.climatology()
        self.ano = {}
        for i, val in enumerate(self.vars):
            ano_tmp = self.dmean[val].rolling(time=21, center=True).mean().dropna('time').groupby('time.dayofyear') - self.clim[val]
            self.ano[val] = ano_tmp.groupby('time.dayofyear') / self.ss[val]
        return self.ano
