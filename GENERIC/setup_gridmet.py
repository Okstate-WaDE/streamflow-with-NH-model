#!/usr/bin/env python

"""
This Python script was used to set up GENERIC, using open-soruce data from online.
"""

__author__ = "Charles Liu"
__date__ = "2025-08-22"

from pathlib import Path
from typing import Tuple
from urllib import request
import os
import json
import statistics
import numpy as np
import pandas as pd
import pygridmet as gridmet
from pynhd import NLDI

# This path assumes that this file, setup_gridmet.py, is located directly in GENERIC.
# If this doesn't work, the paths may need to be modified manually.
DATA_PATH = Path(os.path.dirname(__file__))

# Set up the NLDI database object ahead of time since we only need one instance.
NLDI_DB = NLDI()

def calc_seasonality(attr : pd.DataFrame) -> float:
    """
    Calculates seasonality (a metric for when during the year it's rainiest) from NLDI attributes.

    Parameters
    ----------
    attr : pd.DataFrame
        A Pandas DataFrame containing NLDI attributes that include monthly precipitation averages.
        Other data is optional but unused.
    
    Returns
    -------
    float
        The seasonality.
    """
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    # Multiple linear regression with R^2>0.95 was used to determine the seasonality (since the exact formula
    # for seasonality wasn't explained by Kratzert). bias is the intercept, weights is the coefficients.
    bias = -3.248285727
    weights = [1.240430655, 0, 3.180568668, 3.07540333, 3.562571606, 4.816611353, 6.064900644,
               4.713094804, 4.374751563, 3.31998688, 2.699424697, 1.518576369]
    # Use list comprehension to get a list of monthly precipitation.
    monthly = [attr[f"TOT_PPT7100_{m}"].iloc[0] for m in months]
    total = sum(monthly)
    # Normalize it (important for more accurate regression).
    normalized = [mp/total for mp in monthly]
    for i in range(12):
        bias += weights[i] * normalized[i]
    # Return the seasonality
    return bias

def calc_soil_stats(attr : pd.DataFrame, bias : float, weights : Tuple[float, float, float, float, float, float, float]) -> float:
    """
    Does multiple linear regression (using provided weights and bias) on NLDI attributes.

    Parameters
    ----------
    attr : pd.DataFrame
        A Pandas DataFrame containing NLDI attributes that include soil attributes.
        Other data is optional but unused.
    
    Returns
    -------
    float
        The result of regression.
    """
    # For many of the soil stats, CAMELS did not match NLDI, so I decided to use multiple linear regression
    # to achieve a better result. Some stats were calculated through other means or left as-is.
    result = bias + weights[0] * attr["TOT_SANDAVE"].iloc[0] + weights[1] * attr["TOT_SILTAVE"].iloc[0] \
           + weights[2] * attr["TOT_CLAYAVE"].iloc[0] + weights[3] * attr["TOT_PERMAVE"].iloc[0] \
           + weights[4] * attr["TOT_ROCKDEP"].iloc[0] + weights[5] * attr["TOT_AWCAVE"].iloc[0] \
           + weights[6] * attr["TOT_OLSON_PERM"].iloc[0]
    return result

def calc_prec_stats(df : pd.DataFrame, p_mean : float) -> Tuple[list, list, list, list]:
    """
    Calculates precipitation-related attributes high_prec_freq, high_prec_dur,
    low_prec_freq, and low_prec_dur from meteorological forcings.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas DataFrame containing daily precipitation data.
    p_mean : float
        The average precipitation over all of df.
    
    Returns
    -------
    Tuple[list, list, list, list]
        ([high_prec_freq], [high_prec_dur], [low_prec_freq], [low_prec_dur]). freq is the amount of
        high or low frequency events per year. dur is the average duration of those events. This is
        formatted as a tuple of lists to streamline converting the final data into a DataFrame.
    """
    high_counter = 0 # count number of high precipitation days
    high_events = [] # list of lengths of high precipitation events
    high_streak = 0 # count number of consecutive days in event
    low_counter = 0 # count number of high precipitation days
    low_events = [] # list of lengths of high precipitation events
    low_streak = 0 # count number of consecutive days in event
    for day in df["prcp"]:
        # Check if the precipitation is high.
        if day > 5 * p_mean:
            high_counter += 1
            high_streak += 1
        # If not, check if there is currently a high precipitation event.
        elif high_streak > 0:
            high_events.append(high_streak)
            high_streak = 0
        # Check if the precipitation is low.
        if day < 1:
            low_counter += 1
            low_streak += 1
        # If not, check if there is currently a low precipitation event.
        elif low_streak > 0:
            low_events.append(low_streak)
            low_streak = 0
    # After going through, make sure to end any ongoing events.
    if high_streak > 0:
        high_events.append(high_streak)
    if low_streak > 0:
        low_events.append(low_streak)
    # Return the result in DataFrame-friendly format.
    return ([365 * high_counter/len(df["prcp"])], [statistics.fmean(high_events)],
            [365 * low_counter/len(df["prcp"])], [statistics.fmean(low_events)])

def load_attributes(gauge_id : str, df : pd.DataFrame) -> pd.DataFrame:
    com_id = int(NLDI_DB.getfeature_byid("nwissite", f"USGS-{gauge_id}")["comid"].iloc[0])
    data = {}
    attr = NLDI_DB.get_characteristics(["TOT_SANDAVE", "TOT_SILTAVE", "TOT_CLAYAVE", "TOT_PERMAVE",
                                     "TOT_PPT7100_JAN", "TOT_PPT7100_FEB", "TOT_PPT7100_MAR", "TOT_PPT7100_APR",
                                     "TOT_PPT7100_MAY", "TOT_PPT7100_JUN", "TOT_PPT7100_JUL", "TOT_PPT7100_AUG",
                                     "TOT_PPT7100_SEP", "TOT_PPT7100_OCT", "TOT_PPT7100_NOV", "TOT_PPT7100_DEC",
                                     "TOT_ROCKDEP", "TOT_OLSON_PERM", "TOT_AWCAVE",
                                     "TOT_ELEV_MEAN", "TOT_BASIN_SLOPE", "TOT_BASIN_AREA"], com_id)
    data["p_mean"] = [df["prcp"].mean()]
    data["pet_mean"] = [df["pet"].mean()]
    data["aridity"] = [data["pet_mean"][0] / data["p_mean"][0]]
    data["p_seasonality"] = [calc_seasonality(attr)] # multi regression
    data["frac_snow"] = [df["snow"].mean() / data["p_mean"][0]]
    data["high_prec_freq"], data["high_prec_dur"], data["low_prec_freq"], data["low_prec_dur"] = calc_prec_stats(df, data["p_mean"][0])
    data["elev_mean"] = list(attr["TOT_ELEV_MEAN"])
    data["slope_mean"] = [2.76435 * attr["TOT_BASIN_SLOPE"].iloc[0] - 3.20257] # linear regression
    data["area_gages2"] = list(attr["TOT_BASIN_AREA"])
    data["geol_permeability"] = [np.log10(attr["TOT_PERMAVE"].iloc[0]) - 15]
    data["soil_conductivity"] = [calc_soil_stats(attr, -798.9465898, (8.030292344, 7.968451358, 7.974683175, 0.177809034, 0.003841075, 6.14001913, -0.000990142))]
    data["soil_depth_pelletier"] = [calc_soil_stats(attr, -3152.809045, (31.10022842, 30.9461941, 31.15743186, 1.566249831, 0.550005918, 202.6447689, 0.135290205))]
    data["soil_depth_statsgo"] = [calc_soil_stats(attr, 2.308989753, (-0.021858654, -0.021187821, -0.021052338, 0.005312909, 0.023380569, -0.340289268, -0.000216821))]
    data["sand_frac"] = [calc_soil_stats(attr, -6389.134387, (64.69403656, 63.97348937, 63.85133495, -0.479049078, 0.210056562, -30.94165689, 0.0447348))]
    data["silt_frac"] = [calc_soil_stats(attr, 301.024606, (-3.269011733, -2.264661005, -3.119504075, 0.453361295, 0.324598461, -23.46671657, -0.027876497))]
    data["clay_frac"] = [calc_soil_stats(attr, 632.9354852, (-6.360616113, -6.328565297, -5.609617704, -0.494360363, 0.17123592, -13.96685345, -0.029713411))]
    data["max_water_content"] = [calc_soil_stats(attr, -70.63807025, (0.703842654, 0.705460882, 0.706460986, 0.009564275, 0.01226467, 0.198735965, -0.000384683))]
    data["carbonate_rocks_frac"] = [0.118742623] # average
    data["soil_porosity"] = [calc_soil_stats(attr, -7.621690463, (0.079823683, 0.080530022, 0.080618408, 0.001925408, 0, 0.244896375, -0.0001303))]
    data["frac_forest"] = [0.639539344] # average
    data["lai_max"] = [3.215969887] # average
    data["lai_diff"] = [2.448587614] # average
    data["gvf_max"] = [0.722104301] # average
    data["gvf_diff"] = [0.322748748] # average
    data["gauge_id"] = [gauge_id]
    return pd.DataFrame(data).set_index("gauge_id")

def load_gridmet(geometry : str, dates : Tuple[str, str]) -> pd.DataFrame:
    var = ["pr", "pet", "srad", "tmmx", "tmmn", "sph"] # relative humidity or vpd for calculating vp? also check leap years
    xarr = gridmet.get_bygeom(geometry, dates, variables=var, snow=True)
    xarr = xarr.mean(dim=["lon", "lat"], skipna=True)
    df = xarr.to_dataframe()
    return df

def load_forcings(gauge_id : str) -> Tuple[pd.DataFrame, int]:
    # geometry for get_bygeom
    geometry = NLDI_DB.get_basins(gauge_id).geometry.iloc[0]
    com_id = int(NLDI_DB.getfeature_byid("nwissite", f"USGS-{gauge_id}")["comid"].iloc[0])
    # area in square meters (nldi returns sq km)
    area = int(NLDI_DB.get_characteristics(["TOT_BASIN_AREA"], com_id)["TOT_BASIN_AREA"].iloc[0] * 1000000)
    if gauge_id == "01022500": # this one is weird for some reason
        area = 587675987
    # because there's so much data, do it 5 years at a time
    dfs = []
    for i in range(8):
        dates = (f"{1980+5*i}-01-01", f"{1984+5*i}-12-31")
        df = load_gridmet(geometry, dates)
        print(f"- forcings {i+1}/8 done")
        dfs.append(df)
    # concatenate the segmented data
    df = pd.concat(dfs, axis=0)
    # format data to look more like camels
    df.index.names = ["date"]
    df["prcp"] = df["pr"]
    df["tmax"] = df["tmmx"]
    df["tmin"] = df["tmmn"]
    df["vp"] = (101325 * df["sph"]) / (0.622 + 0.378 * df["sph"])
    df = df.drop(["spatial_ref", "pr", "tmmx", "tmmn", "sph"], axis=1)
    return (df, area)

def load_streamflow(gauge_id : str, area : int) -> pd.Series:
    # open from online
    parameters = "agencyCd=USGS&statCd=00003&parameterCd=00060&startDT=1980-01-01&endDT=2019-12-31"
    usgs_url = f"https://waterservices.usgs.gov/nwis/dv?format=json&siteStatus=all&site={gauge_id}&{parameters}"
    with request.urlopen(usgs_url) as f:
        usgs = f.read().decode('utf-8')
        jdb = json.loads(usgs)
    timeseries = jdb["value"]["timeSeries"][0]["values"][0]["value"]
    df = pd.DataFrame.from_dict(timeseries)
    df["date"] = pd.to_datetime(df["dateTime"])
    df = df.set_index("date")
    df["QObs"] = pd.to_numeric(df["value"])
    # replace invalid values with NaN
    df.loc[df["QObs"] < 0, "QObs"] = np.nan
    # normalize discharge from cubic feet per second to mm per day
    df["QObs"] = 28316846.592 * df["QObs"] * 86400 / (area * 10**6)
    return df["QObs"]

# get basin list
with open(os.path.join(DATA_PATH, "basin_list.txt"), "r") as list_file:
    basin_list = list_file.read().splitlines()
    # should be a list of gauge_id

for basin in basin_list:
    print(f"loading data for basin {basin}...")
    attributes_path = os.path.join(DATA_PATH, f"attributes/{basin}.csv")
    timeseries_path = os.path.join(DATA_PATH, f"time_series/{basin}.nc4")
    if os.path.exists(attributes_path) and os.path.exists(timeseries_path):
        print("- already exists")
        continue
    # get daymet data and process it for attributes
    forcings, area = load_forcings(basin)
    attributes = load_attributes(basin, forcings)
    attributes.to_csv(attributes_path)
    print(f"- attributes saved to {basin}.csv")
    # re-index because daymet does weird stuff on leap years
    forcings = forcings.reindex(pd.date_range('01-01-1980', '12-31-2019'))
    forcings.index.names = ["date"]
    forcings = forcings.drop(["snow", "pet"], axis=1)
    # combine (duplicate) different forcings
    daymet_df = forcings.rename({"prcp": "prcp(mmperday)_daymet", "srad": "srad(Wperm2)_daymet", "tmax": "tmax(C)_daymet",
                       "tmin": "tmin(C)_daymet", "vp": "vp(Pa)_daymet"}, axis=1)
    maurer_df = forcings.rename({"prcp": "PRCP(mmperday)_maurer", "srad": "SRAD(Wperm2)_maurer", "tmax": "Tmax(C)_maurer",
                        "tmin": "Tmin(C)_maurer", "vp": "Vp(Pa)_maurer"}, axis=1)
    nldas_df = forcings.rename({"prcp": "PRCP(mmperday)_nldas", "srad": "SRAD(Wperm2)_nldas", "tmax": "Tmax(C)_nldas",
                        "tmin": "Tmin(C)_nldas", "vp": "Vp(Pa)_nldas"}, axis=1)
    dfs = [daymet_df, maurer_df, nldas_df]
    df = pd.concat(dfs, axis=1)
    # add streamflow data
    df["QObs(mmperd)"] = load_streamflow(basin, area)
    print("- streamflow done")
    # save it to netcdf
    xarr = df.to_xarray()
    xarr.to_netcdf(timeseries_path)
    print(f"- forcings saved to {basin}.nc4")