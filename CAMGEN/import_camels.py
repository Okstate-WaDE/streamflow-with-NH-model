#!/usr/bin/env python

"""
This Python script was used to create the time_series folder for CAMGEN, using the data from CAMELS.
If CAMGEN is already downloaded, running this should do nothing.
"""

__author__ = "Charles Liu"
__date__ = "08-18-2025"

from pathlib import Path
from typing import Tuple
import os
import numpy as np
import pandas as pd

# These paths assume that this file, import_time_series.py, is located directly in CAMGEN,
# that CAMGEN and CAMELS are both located in the same directory, and that CAMELS is formatted correctly.
# If this doesn't work, the paths may need to be modified manually.
camgen_path = Path(os.path.dirname(__file__))
camels_path = Path(os.path.join(camgen_path, "../CAMELS"))

def load_attributes(group : str) -> list:
    """
    Loads attributes from CAMELS .txt format and writes into CAMGEN .csv format.

    Parameters
    ----------
    group : str
        One of "clim", "geol", "hydro", "name", "soil", "topo", and "vege". The group of attributes to load.
    
    Returns
    -------
    list
        A list of all basins that attributes are stored for. Useful for forcings and streamflow later.
    """
    src_path = os.path.join(camels_path, f"camels_attributes_v2.0/camels_{group}.txt")
    dst_path = os.path.join(camgen_path, f"attributes/{group}.csv")
    # Open CAMELS static attributes.
    with open(src_path, "r") as file:
        content = file.read()
    # Remove commas and replace semicolons with commas.
    content = content.replace(",", "").replace(";", ",")
    with open(dst_path, "w") as file:
        file.write(content)
    # Open new file as a CSV and get the basin list (also check that it works).
    df = pd.read_csv(dst_path, dtype={0: str})
    return list(df["gauge_id"])

def load_forcings(basin : str, forcing : str) -> Tuple[pd.DataFrame, int]:
    """
    Loads forcings from CAMELS .txt format into a Pandas DataFrame.

    Parameters
    ----------
    basin : str
        The USGS gauge ID of the basin to load.
    forcing : str
        One of "daymet", "maurer", or "nldas". The specific forcing to load for the basin.
    
    Returns
    -------
    Tuple[pd.DataFrame, int]
        A Pandas DataFrame containing the CAMELS data for this basin and forcing pair.
        The area of the basin, in square meters.
    """
    # Using the Path class's .glob method, locate any files that might contain the data.
    file_path = list(camels_path.glob(f"basin_mean_forcing/{forcing}/*/{basin}_*_forcing_leap.txt"))
    if file_path:
        # If files containing data exist, pick the first one.
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f"No file for Basin {basin} at {file_path}")
    # Open the .txt file and skip the metadata.
    with open(file_path, "r") as fp:
        # First few lines unneeded.
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        # Used raw string to denote whitespace character, syntax error otherwise.
        df = pd.read_csv(fp, sep=r"\s+")
    # Create a new column to index the DataFrame chronlogically.
    df["date"] = pd.to_datetime(df["Year"].map(str) + "/" + df["Mnth"].map(str) + "/" + df["Day"].map(str), format="%Y/%m/%d")
    df = df.set_index("date")
    # NetCDF files don't like the "/" character, so replace it with "per". Also, append forcing name for organization.
    df = df.rename(columns={col: f"{col.replace("/", "per")}_{forcing}" for col in df.columns})
    return df, area

def load_streamflow(basin : str, area : int) -> pd.Series:
    """
    Loads observed streamflow data from CAMELS .txt format into a Pandas Series.

    Parameters
    ----------
    basin : str
        The USGS gauge ID of the basin to load.
    area : int
        The area of the basin, in square meters. Necessary for converting units.
    
    Returns
    -------
    pd.Series
        A Pandas Series indexed by DateTime, containing streamflow data.
    """
    # Using the Path class's .glob method, locate any files that might contain the data.
    file_path = list(camels_path.glob(f"usgs_streamflow/*/{basin}_streamflow_qc.txt"))
    if file_path:
        # If files containing data exist, pick the first one.
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f"No file for Basin {basin} at {file_path}")
    # CAMELS streamflow files don't contain a header with column labels, so insert that manually.
    col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
    # Create a new column to index the DataFrame chronlogically.
    df["date"] = pd.to_datetime(df["Year"].map(str) + "/" + df["Mnth"].map(str) + "/" + df["Day"].map(str), format="%Y/%m/%d")
    df = df.set_index("date")
    # Replace invalid values with NaN.
    df.loc[df["QObs"] < 0, "QObs"] = np.nan
    # Normalize discharge from cubic feet per second to mm per day.
    df["QObs"] = 28316846.592 * df["QObs"] * 86400 / (area * 10**6)
    return df["QObs"]

# Set up these parameters in advance.
groups = ["clim", "geol", "hydro", "name", "soil", "topo", "vege"]
forcings = ["daymet", "maurer", "nldas"]

# Load attributes.
for group in groups:
    basin_list = load_attributes(group)

for basin in basin_list:
    # Check if the basin already has data.
    if os.path.exists(os.path.join(camgen_path, f"time_series/{basin}.nc4")):
        continue
    # Combine forcings.
    dfs = []
    for forcing in forcings:
        df, area = load_forcings(basin, forcing)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    # Add the observed streamflow data to the DataFrame.
    df["QObs(mmperd)"] = load_streamflow(basin, area)
    # Save as a NetCDF file.
    file_path = os.path.join(camgen_path, f"time_series/{basin}.nc4")
    xarr = df.to_xarray()
    xarr.to_netcdf(file_path)