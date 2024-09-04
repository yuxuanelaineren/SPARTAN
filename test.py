#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # cartopy must be >=0.19
import xarray as xr
import cartopy.feature as cfeature
import pandas as pd
from gamap_colormap import WhGrYlRd
import calendar
import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}-{}/monthly/'.format(cres.lower(), deposition) # CEDS, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower()) # HTAP, LUO
# sim_dir = '/Volumes/rvmartin/Active/dandan.z/AnalData/WUCR3-C360/' # EDGAR, LUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/WUCR3-C360/' # EDGAR, LUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year) # C720, HTAP, LUO
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
################################################################################################
# Combine SPARTAN and GCHP dataset based on lat/lon
################################################################################################
# Loop through each month
for mon in range(1, 13):
    # Load simulated data
    with xr.open_dataset(sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon),engine='netcdf4') as sim_df:  # CEDS
        conc = sim_df[species].values
    print(conc.shape)
    # Load population data
    pop_df = xr.open_dataset(support_dir + 'Regrid.PopDen.latlon.1800x3600.to.{}.conserve.2015.nc4'.format(cres.upper())).squeeze()
    pop = pop_df['pop'].values
    print(pop.shape)
    # # Mask out sea areas
    # lsmask_df = xr.open_dataset(support_dir + 'Regrid.LL.1800x3600.{}.neareststod.landseamask.nc'.format(cres.upper())).squeeze()
    # lsmask = lsmask_df['mask'].values
    # land_mask = lsmask < 50  # < 50 represents land
    # conc = np.where(land_mask, conc, np.nan)
    # pop = np.where(land_mask, pop, np.nan)

    conc = conc.flatten()
    pop = pop.flatten()
    # Population-weighted conc (pwm)
    ind = np.where(~np.isnan(conc))
    N = len(conc[ind])
    pwm = np.nansum(pop[ind] * conc[ind]) / np.nansum(pop[ind]) # compute pwm, one value
    pwstd = np.sqrt(np.nansum(pop[ind] * (conc[ind] - pwm) ** 2) / ((N - 1) / N * np.nansum(pop[ind])))
    pwse = pwstd / np.sqrt(N)
    print(f"Population-weighted mean (pwm): {pwm}")
    print(f"Population-weighted std (pwstd): {pwstd}")
    print(f"Population-weighted se (pwse): {pwse}")