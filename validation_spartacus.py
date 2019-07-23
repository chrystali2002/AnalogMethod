import xarray as xr
import numpy as np
import dask
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import iris
from eofs.iris import Eof
from eofs.multivariate.iris import MultivariateEof
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import datetime
import time
import cloudpickle

#path = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/AnalogMethod/'
path = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/'

#%%
# Load analoga data.
analoga = cloudpickle.load( open( path + "AnalogMethod/analoga_list_01-01_12-31.p", "rb" ) )

#%%
# Define start and end day of validation period.
start_day = '1990-01-01'
end_day = '1990-12-31'
param = 'RR'

# Find index of start and end day.
start_day_occurences = np.where(analoga == datetime.datetime.strptime(start_day, '%Y-%m-%d'))
start_day_idx = start_day_occurences[0][np.where(start_day_occurences[1] == 0)][0]

end_day_occurences = np.where(analoga == datetime.datetime.strptime(end_day, '%Y-%m-%d'))
end_day_idx = end_day_occurences[0][np.where(end_day_occurences[1] == 0)][0]

#%%
# Load Spartacus data (domain: Austria).
data_sparta = xr.open_mfdataset(path+'/sparta/' + param + '/*.nc', chunks={'time': 5000})#, 'lat': 141, 'lon': 141})

#%%
# Define number of analoga to be verified.
no_analoga = 5

# Create arrays for RMSE and Correlation Coefficient.
arr_rmse = np.zeros((len(data_sparta.time), no_analoga+1), dtype='float')
arr_rmse[:,0] = range(0,len(data_sparta.time))
arr_corr = arr_rmse.copy()

# Select period.
arr_rmse = arr_rmse[start_day_idx:end_day_idx+1]
arr_corr = arr_corr[start_day_idx:end_day_idx+1]

#%%
# Time script duration.
t_start = time.time()

# Validation.
# Loop over period to be verified.
for i, list_day in enumerate(analoga[start_day_idx:end_day_idx+1]):
    # Select reference day (first day in array)
    ref_day = data_sparta.sel(time = list_day[0])

    # Remove all nan values (outside shape of Austria)
    arr_ref_day = ref_day.RR.values[~np.isnan(ref_day.RR.values)]

    # Loop over analoga for reference day.
    for j,analogon in enumerate(list_day[1:]):
        # Select j_th analogon.
        try:
            analog_day = data_sparta.sel(time = analogon)
        
            # Remove all nan values (outside shape of Austria)
            arr_analog_day = analog_day.RR.values[~np.isnan(analog_day.RR.values)]
    
            # Calculate RMSE and add to RMSE array.
            rmse = np.sqrt( ( arr_analog_day - arr_ref_day )**2 ).mean()
            arr_rmse[i][j+1] = rmse
    
            # Calculate Correlation Coefficient and add to Correlation Coefficient array.
            corr = np.corrcoef(arr_ref_day, arr_analog_day)[0][1]
            arr_corr[i][j+1] = corr
        
        except KeyError:
            arr_rmse[i][j+1] = np.nan
            arr_corr[i][j+1] = np.nan
            print(analogon, ' does not exist in Spartacus data.')
            
    t_end_tmp = time.time()
    print("total time after " + str(list_day[0]) + ": ", \
          str(datetime.timedelta(seconds=t_end_tmp - t_start)), '\n')

# Save RMSE and Correlation Coefficient array as pickle.
print("Saving rmse array to " + path + "rmse_arr_" + start_day + '_' + end_day + ".p")
cloudpickle.dump( arr_rmse, open( path + "rmse_arr_" + start_day + '_' + end_day + ".p", "wb" ) )

print("Saving correlation coefficient array to " + path + "corr_arr_" + start_day + '_' + end_day + ".p")
cloudpickle.dump( arr_corr, open( path + "corr_arr_" + start_day + '_' + end_day + ".p", "wb" ) )
#%%
# Load rmse and corr array
arr_rmse = cloudpickle.load( open( path + "/AnalogMethod/" + param + "_rmse_arr_" + start_day + '_' + end_day + ".p", "rb" ) )
arr_corr = cloudpickle.load( open( path + "/AnalogMethod/" + param + "_corr_arr_" + start_day + '_' + end_day + ".p", "rb" ) )

#%%
# Plotting.
savefig = 1

start = datetime.datetime.strptime(start_day, "%Y-%m-%d")
end = datetime.datetime.strptime(end_day, "%Y-%m-%d")
date_days = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+1)]

days = arr_rmse[:,0]

rmse_ana_1 = arr_rmse[:,1]
rmse_ana_2 = arr_rmse[:,2]
rmse_ana_3 = arr_rmse[:,3]
rmse_ana_4 = arr_rmse[:,4]
rmse_ana_5 = arr_rmse[:,5]
list_rmse_ana = [rmse_ana_1, rmse_ana_2, rmse_ana_3, rmse_ana_4, rmse_ana_5]

corr_ana_1 = arr_corr[:,1]
corr_ana_2 = arr_corr[:,2]
corr_ana_3 = arr_corr[:,3]
corr_ana_4 = arr_corr[:,4]
corr_ana_5 = arr_corr[:,5]
list_corr_ana = [corr_ana_1, corr_ana_2, corr_ana_3, corr_ana_4, corr_ana_5]


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
color_list = ['r', 'b', 'g', 'y', 'k']

for i in range(0,5):
    if i == 0:
        ax1.plot(date_days, list_rmse_ana[i], color=color_list[i], 
                 label='analogon_'+str(i+1), linewidth=2, zorder=4)
        ax2.plot(date_days, list_corr_ana[i], color=color_list[i], 
                 label='analogon_'+str(i+1), linewidth=2, zorder=4)
    else:
        ax1.plot(date_days, list_rmse_ana[i], color=color_list[i], 
                 label='analogon_'+str(i+1), alpha=0.5)
        ax2.plot(date_days, list_corr_ana[i], color=color_list[i], 
                 label='analogon_'+str(i+1), alpha=0.5)

months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%Y-%m')

for ax in [ax1, ax2]:
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.set_xlabel('time')

ax1.set_title('RMSE for first ' + str(no_analoga) + 
              ' Analoga for parameter ' + param + 
              ' from ' + start_day + ' until ' + end_day, fontsize=12)
ax2.set_title('Correlation Coefficient for first ' + str(no_analoga) + 
              ' Analoga for parameter ' + param + 
              ' from ' + start_day + ' until ' + end_day, fontsize=12)

ax1.set_ylabel('RMSE')
ax2.set_ylabel('Correlation Coefficient')

ax1.legend(loc=1)
ax2.legend(loc=3)
plt.subplots_adjust(hspace=0.3)


if savefig == 1:
    plt.savefig(path + '/' + param + '_' + start_day + '_' + end_day + '.png')

#%%
for i,rmse in enumerate(list_rmse_ana):
    print('rmse for analogon_' + str(i+1) + ': ', np.nanmedian(rmse))

for i,corr in enumerate(list_corr_ana):
    print('corr for analogon_' + str(i+1) + ': ', np.nanmedian(corr))

#%%
eof = cloudpickle.load( open( path + "eof_msl_list_01-01_12-31.p", "rb" ) )
print(len(eof))
pc = cloudpickle.load( open( path + "pc_list_01-01_12-31.p", "rb" ) )
print(len(pc))

#%%
savefig = 1

for day in [0,90,181,272]:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, 
     figsize=(15,4), subplot_kw={'projection': ccrs.PlateCarree()})
    
    for idx,ax_tmp in enumerate([ax1, ax2, ax3]):
        #ax_tmp = plt.axes(projection=ccrs.PlateCarree())
        #ax_tmp.axes(projection=ccrs.PlateCarree())
        ax_tmp.set_extent([-10, 25, 32.5, 67])
        ax_tmp.coastlines()
        #ax.add_feature(cfeature.LAND)
        
        #ax.contourf(eof[0][0].lon, eof[0][0].lat, eof[0][0].values, alpha=0.5)
        eof[day][idx].plot(ax=ax_tmp, transform=ccrs.PlateCarree(),
                 cbar_kwargs={'shrink': 0.4})

    plt.suptitle('First 3 EOF of param MSLP on day ' + str(day+1) + ' of the year')
    if savefig == 1:
        plt.savefig(path + '/EOF_MSLP_DayNo_' + str(day) + '.png')
 
