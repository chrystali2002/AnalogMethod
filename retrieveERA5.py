#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 06:59:28 2019

@author: martinschneider
"""

import pandas as pd
import cdsapi

### Configuration for Download
year_start = 1979
year_end = 2018
SAVE_DIR = '/Users/martinschneider/Documents/Studium_Meteorologie/Master/4.Semester/Klimamodellierung/Projektarbeit/ERA5/'
ERA5_dataset = 'reanalysis-era5-single-levels'
param = ['mean_sea_level_pressure']
p_level = ''
filename = 'mslp'
#ERA5_dataset = 'reanalysis-era5-pressure-levels'
#param = ['relative_humidity','specific_humidity']
#p_level = '700'
#filename = 'humidity'


### Functions
def retrieveERAinterim(datelist):
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()
    server.retrieve({
        "class": "ei",
        "format": "netcdf",
        "dataset": "interim",
        # datelist format: "YYYYMMDD/YYYYMMDD" 
        # e.g. 19940101/19940201/19940301/19940401/19940501/"
        "date": datelist,
        "expver": "1",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": "151.128/165.128/166.128/167.128",
        "stream": "moda",
        "type": "an",
        "target": "output",
    })
    return

def retrieveERA5(year_start,year_end,ERA5_dataset,param,p_level,SAVE_DIR):
    for year in range(year_start, year_end):
        for mon in range(1,12):
            if mon < 10:
                month = '0'+str(mon)
            else:
                month = str(mon)
            c = cdsapi.Client()
            r = c.retrieve(
                           ERA5_dataset, {
                           'pressure_level':p_level,
                           'variable':param,
                           'product_type': 'reanalysis',
                           'year'        : str(year),
                           'month'       : month,
                           # ['01', '02', '03', '04',
                           # '05', '06', '07', '08',
                           # '09', '10', '11', '12'],
                           'day': ['01', '02', '03', '04',
                                   '05', '06', '07', '08',
                                   '09', '10', '11', '12',
                                   '13', '14', '15', '16',
                                   '17', '18', '19', '20',
                                   '21', '22', '23', '24',
                                   '25', '26', '27', '28',
                                   '29', '30', '31'],
                           'time'        : [
                                            '00:00','01:00','02:00',
                                            '03:00','04:00','05:00',
                                            '06:00','07:00','08:00',
                                            '09:00','10:00','11:00',
                                            '12:00','13:00','14:00',
                                            '15:00','16:00','17:00',
                                            '18:00','19:00','20:00',
                                            '21:00','22:00','23:00'
                                            ],
                           'area'        : [67.5, -10, 32.5, 25], # North, West, South, East.
                           'format'      : 'netcdf'
                           })
            r.download(SAVE_DIR + '/' + filename + '_'+str(year)+month+'.nc')


def createDatelistStr(start,end):
    # start, end format:
    # 'YYYY-mm-dd' e.g. '2014-01-01'
    datelist = pd.date_range(start,end, 
              freq='MS').strftime("%Y%m%d").tolist()
    str1 = '/'.join(datelist)
    return datelist
    #date1994 = "19940101/19940201/19940301/19940401/19940501/19940601/19940701/19940801/19940901/19941001/19941101/19941201"

#start = '1979-01-01'
#end = '2018-12-31'
#datelist = createDatelistStr(start,end)
#retrieveERAinterim(datelist)

### Retireve ERA5 data
retrieveERA5(year_start,year_end,ERA5_dataset,param,p_level,SAVE_DIR)


