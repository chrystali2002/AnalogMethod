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

for _,ds in ano_roll.groupby('time.dayofyear'):
        ds = ds.rename({'time':'time_old'})
        ds = ds.stack(time=['time_old', 'window_dim']).transpose('time','latitude','longitude')
#        ds = ds.assign_coords(time=ds.time_old)
#        ds = ds.drop('dayofyear')
        print(ds.r.time)
        # Convert DataArrays of individual variables to iris cubes
        # (required for multivariate EOF)        
        ds_iris_r = ds.r.to_iris()
        ds_iris_q = ds.q.to_iris()
        cube_list = iris.cube.CubeList([ds_iris_r, ds_iris_q])
        ds_iris = cube_list.merge()
#        for coord in ds_iris_r.coords():
#            print(coord.name())
#        print(ds_iris_r)
        
        # Calculate multivariate EOF
        solver = MultivariateEof([ds_iris_r, ds_iris_q], weights=None)
        eof = solver.eofs(neofs=n)
        pc = solver.pcs(npcs=n, pcscaling=1)
        eigval = solver.eigenvalues(neigs=n)
        eof_var = solver.varianceFraction(neigs=n)
        pseudo_pc = solver.projectField(new_cube, neofs=n)
        
        # Convert to xarray
        ds_eof = xr.DataArray.from_iris(eof)
        print(ds_eof)
