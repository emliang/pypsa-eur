# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build hydroelectric inflow time-series for each country using custom ERA5 inflow data.

This is a modified version of build_hydro_profile.py that works with 
0.25° × 0.25° gridded ERA5 surface inflow data.
"""

import logging

import country_converter as coco
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from numpy.polynomial import Polynomial

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    set_scenario_config,
)

cc = coco.CountryConverter()
logger = logging.getLogger(__name__)


def load_custom_era5_inflow(file_path: str, time_slice=None) -> xr.Dataset:
    """
    Load custom ERA5 inflow data.
    
    Parameters
    ----------
    file_path : str
        Path to your ERA5 inflow NetCDF file
    time_slice : slice, optional
        Time slice to select from the data
        
    Returns
    -------
    xr.Dataset
        Inflow data with standardized coordinates
    """
    # Load your ERA5 inflow data
    ds = xr.open_dataset(file_path)
    
    # Standardize coordinate names (adjust based on your file structure)
    coord_mapping = {
        'longitude': 'x',
        'latitude': 'y', 
        'lon': 'x',
        'lat': 'y'
    }
    
    for old_name, new_name in coord_mapping.items():
        if old_name in ds.coords:
            ds = ds.rename({old_name: new_name})
    
    # If you have a different variable name for inflow, rename it to 'runoff'
    # Adjust this based on your actual variable name
    if 'surface_runoff' in ds.data_vars:
        ds = ds.rename({'surface_runoff': 'runoff'})
    elif 'inflow' in ds.data_vars:
        ds = ds.rename({'inflow': 'runoff'})
    
    # Select time slice if provided
    if time_slice is not None:
        ds = ds.sel(time=time_slice)
    
    return ds


def aggregate_inflow_to_countries(
    inflow_data: xr.Dataset, 
    country_shapes: gpd.GeoSeries
) -> xr.DataArray:
    """
    Aggregate gridded inflow data to country level.
    
    Parameters
    ----------
    inflow_data : xr.Dataset
        Gridded inflow data
    country_shapes : gpd.GeoSeries
        Country geometries
        
    Returns
    -------
    xr.DataArray
        Country-aggregated inflow profiles
    """
    # Create a simple grid for spatial aggregation
    lons = inflow_data.coords['x'].values
    lats = inflow_data.coords['y'].values
    
    # Create grid points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Flatten for easier processing
    points = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten()
    })
    
    # Create GeoDataFrame of grid points
    from shapely.geometry import Point
    geometry = [Point(lon, lat) for lon, lat in zip(points.lon, points.lat)]
    grid_points = gpd.GeoDataFrame(points, geometry=geometry, crs='EPSG:4326')
    
    # Spatial join to assign grid points to countries
    joined = gpd.sjoin(grid_points, country_shapes.to_frame('geometry'), how='left')
    
    # Calculate country-level aggregated inflow
    country_inflow = {}
    
    for country in country_shapes.index:
        # Get grid points in this country
        country_mask = joined.index_right == country
        country_points = joined[country_mask]
        
        if len(country_points) > 0:
            # Get corresponding grid indices
            lon_indices = [np.argmin(np.abs(lons - lon)) for lon in country_points.lon]
            lat_indices = [np.argmin(np.abs(lats - lat)) for lat in country_points.lat]
            
            # Extract inflow data for these points and sum
            country_total = 0
            for lon_idx, lat_idx in zip(lon_indices, lat_indices):
                country_total += inflow_data.runoff.isel(x=lon_idx, y=lat_idx)
            
            country_inflow[country] = country_total
        else:
            # No grid points in country, set to zero
            country_inflow[country] = xr.zeros_like(inflow_data.runoff.isel(x=0, y=0))
    
    # Combine into single DataArray
    country_da = xr.concat(
        [country_inflow[c] for c in country_shapes.index],
        dim=pd.Index(country_shapes.index, name='countries')
    )
    
    return country_da


def get_eia_annual_hydro_generation(
    fn: str, countries: list[str], capacities: bool = False
) -> pd.DataFrame:
    # Same as original function - keeping for EIA data normalization
    df = pd.read_csv(fn, skiprows=2, index_col=1, na_values=[" ", "--"]).iloc[1:, 1:]
    df.index = df.index.str.strip()
    df.columns = df.columns.astype(int)

    former_countries = {
        "Former Czechoslovakia": dict(
            countries=["Czechia", "Slovakia"], start=1980, end=1992
        ),
        "Former Serbia and Montenegro": dict(
            countries=["Serbia", "Montenegro", "Kosovo"], start=1992, end=2005
        ),
        "Former Yugoslavia": dict(
            countries=[
                "Slovenia",
                "Croatia",
                "Bosnia and Herzegovina",
                "Serbia",
                "Kosovo",
                "Montenegro",
                "North Macedonia",
            ],
            start=1980,
            end=1991,
        ),
    }

    for k, v in former_countries.items():
        period = [i for i in range(v["start"], v["end"] + 1)]
        ratio = df.loc[v["countries"]].T.dropna().sum()
        ratio /= ratio.sum()
        for country in v["countries"]:
            df.loc[country, period] = df.loc[k, period] * ratio[country]

    baltic_states = ["Latvia", "Estonia", "Lithuania"]
    df.loc[baltic_states] = (
        df.loc[baltic_states].T.fillna(df.loc[baltic_states].mean(axis=1)).T
    )

    df.loc["Germany"] = df.filter(like="Germany", axis=0).sum()
    df = df.loc[~df.index.str.contains("Former")]
    df.drop(["Europe", "Germany, West", "Germany, East"], inplace=True)

    df.index = cc.convert(df.index, to="iso2")
    df.index.name = "countries"

    # convert to MW or MWh/a
    factor = 1e3 if capacities else 1e6
    df = df.T[countries] * factor

    df.ffill(axis=0, inplace=True)

    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_hydro_profile_custom")
    
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params_hydro = snakemake.params.hydro
    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    countries = snakemake.params.countries

    # Load country shapes
    country_shapes = (
        gpd.read_file(snakemake.input.country_shapes)
        .set_index("name")["geometry"]
        .reindex(countries)
    )
    country_shapes.index.name = "countries"

    # Load your custom ERA5 inflow data
    logger.info("Loading custom ERA5 inflow data...")
    inflow_data = load_custom_era5_inflow(
        snakemake.input.custom_era5_inflow, 
        time_slice=time
    )

    # Aggregate to country level
    logger.info("Aggregating inflow to country level...")
    inflow = aggregate_inflow_to_countries(inflow_data, country_shapes)

    # Optional: Normalize using EIA statistics (if available)
    if hasattr(snakemake.input, 'eia_hydro_generation'):
        logger.info("Normalizing using EIA statistics...")
        eia_stats = get_eia_annual_hydro_generation(
            snakemake.input.eia_hydro_generation, countries
        )
        
        # Simple normalization by annual totals
        years_in_time = pd.DatetimeIndex(time).year.unique()
        for year in years_in_time:
            if year in eia_stats.index:
                year_mask = pd.DatetimeIndex(inflow.time.values).year == year
                inflow_year = inflow.sel(time=year_mask)
                annual_total = inflow_year.sum('time')
                
                # Normalize to EIA statistics
                for country in countries:
                    if country in eia_stats.columns and annual_total.sel(countries=country) > 0:
                        scaling_factor = eia_stats.loc[year, country] / annual_total.sel(countries=country)
                        inflow.loc[dict(countries=country, time=year_mask)] *= scaling_factor

    # Apply minimum inflow clipping
    if "clip_min_inflow" in params_hydro:
        inflow = inflow.where(inflow > params_hydro["clip_min_inflow"], 0)

    # Convert units if needed (your data might be in different units)
    # Adjust this conversion based on your data units
    # inflow = inflow * conversion_factor  # e.g., m/day to MW

    # Save output
    inflow.to_netcdf(snakemake.output.profile)
    logger.info(f"Hydro profile saved to {snakemake.output.profile}")
