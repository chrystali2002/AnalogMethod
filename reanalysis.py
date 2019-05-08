import xarray as xr
import dask

class GCM_data:
    def __init__(self, datapath):
        self.datapath = datapath
    
    def create_xarray(self):
        self.ds = xr.open_mfdataset(self.datapath+'*.nc', chunks={'time': 504, 
                                                                  'latitude': 141, 'longitude': 141})
        return self.ds
    
    def convert_to_dailymean(self):
        self.ds = self.create_xarray()
        self.daily_mean = self.ds['r'].resample(time='1D').mean(axis=0).chunk({'time': 42, 
                                                                          'longitude': 141, 'latitude':141})
        return self.daily_mean
    
    def climatology(self):
        self.r_daily = self.convert_to_dailymean()
        self.r_clim = self.r_daily.rolling(time=21, center=True).mean().groupby('time.day').mean(axis=0)
        return self.r_clim
