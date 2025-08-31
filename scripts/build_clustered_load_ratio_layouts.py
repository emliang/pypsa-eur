# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build load distribution ratio layouts for all clustered model regions based on 
population and GDP data. This combines the load distribution logic from 
build_electricity_demand_base.py with the clustering approach from 
build_clustered_population_layouts.py.

The script calculates load distribution ratios that can be used to distribute
electricity demand across clustered regions based on weighted GDP and population data.
"""

import logging
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import xarray as xr
from shapely.prepared import prep

from scripts._helpers import configure_logging, load_cutout, set_scenario_config

logger = logging.getLogger(__name__)


def normed(s: pd.Series) -> pd.Series:
    """Normalize a pandas Series to sum to 1.0"""
    return s / s.sum()


def shapes_to_shapes(orig: gpd.GeoSeries, dest: gpd.GeoSeries) -> sparse.lil_matrix:
    """
    Create a sparse matrix for transferring data between geometries.
    Adopted from vresutils.transfer.Shapes2Shapes()
    
    Parameters
    ----------
    orig : gpd.GeoSeries
        Source geometries
    dest : gpd.GeoSeries  
        Destination geometries
        
    Returns
    -------
    sparse.lil_matrix
        Transfer matrix with shape (len(dest), len(orig))
    """
    orig_prepped = list(map(prep, orig))
    transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=float)

    for i, j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest.iloc[i]):
            area = orig.iloc[j].intersection(dest.iloc[i]).area
            transfer[i, j] = area / dest.iloc[i].area

    return transfer


def calculate_load_distribution_factors(
    clustered_regions: gpd.GeoDataFrame,
    nuts3_shapes: gpd.GeoDataFrame,
    distribution_key: dict[str, float],
) -> pd.DataFrame:
    """
    Calculate load distribution factors for clustered regions based on GDP and population.
    
    Parameters
    ----------
    clustered_regions : gpd.GeoDataFrame
        Clustered regions with geometry
    nuts3_shapes : gpd.GeoDataFrame
        NUTS3 regions with GDP and population data
    distribution_key : dict
        Weights for GDP and population in distribution calculation
        
    Returns
    -------
    pd.DataFrame
        Load distribution factors for each clustered region by country
    """
    gdp_weight = distribution_key.get("gdp", 0.6)
    pop_weight = distribution_key.get("population", distribution_key.get("pop", 0.4))
    
    load_factors = {}
    
    for country, group in clustered_regions.groupby('country'):
        logger.info(f"Processing load factors for country: {country}")
        
        if len(group) == 1:
            # Single region for this country - factor is 1.0
            factors = pd.Series(1.0, index=group.index)
        else:
            # Multiple regions - calculate based on GDP and population
            nuts3_country = nuts3_shapes[nuts3_shapes.country == country]
            
            if nuts3_country.empty:
                logger.warning(f"No NUTS3 data found for country {country}, using equal distribution")
                factors = pd.Series(1.0 / len(group), index=group.index)
            else:
                # Calculate transfer matrix from clustered regions to NUTS3
                transfer = shapes_to_shapes(group.geometry, nuts3_country.geometry).T.tocsr()
                
                # Aggregate GDP and population data to clustered regions
                gdp_clustered = pd.Series(
                    transfer.dot(nuts3_country["gdp"].fillna(1.0).values), 
                    index=group.index
                )
                pop_clustered = pd.Series(
                    transfer.dot(nuts3_country["pop"].fillna(1.0).values), 
                    index=group.index
                )
                
                # Calculate weighted factors
                factors = normed(
                    gdp_weight * normed(gdp_clustered) + pop_weight * normed(pop_clustered)
                )
        
        load_factors[country] = factors
    
    return pd.concat(load_factors, names=['country', 'region'])


def build_clustered_load_ratios(
    clustered_regions_file: str,
    pop_layout_files: dict,
    nuts3_shapes_file: str,
    cutout,
    distribution_key: dict[str, float],
) -> pd.DataFrame:
    """
    Build clustered load ratio layouts combining population data and load distribution logic.
    
    Parameters
    ----------
    clustered_regions_file : str
        Path to clustered regions geojson file
    pop_layout_files : dict
        Dictionary with paths to population layout files (total, urban, rural)
    nuts3_shapes_file : str
        Path to NUTS3 shapes file with GDP/population data
    cutout : atlite.Cutout
        Atlite cutout for spatial operations
    distribution_key : dict
        Weights for GDP and population distribution
        
    Returns
    -------
    pd.DataFrame
        Load ratios and population data for clustered regions
    """
    # Load clustered regions
    clustered_regions_gdf = gpd.read_file(clustered_regions_file).set_index("name")
    
    # Apply buffer operation to geometry and create GeoSeries for indicator matrix
    clustered_regions_buffered = clustered_regions_gdf.geometry.buffer(0)
    
    # Extract country code from region name (first 2 characters) 
    clustered_regions_gdf["country"] = clustered_regions_gdf.index.str[:2]
    
    # Load NUTS3 shapes with GDP/population data
    nuts3_shapes = gpd.read_file(nuts3_shapes_file).set_index("index")
    
    # Calculate indicator matrix for population aggregation
    I = cutout.indicatormatrix(clustered_regions_buffered)  # noqa: E741
    
    # Aggregate population data to clustered regions
    pop = {}
    for item in ["total", "urban", "rural"]:
        pop_layout = xr.open_dataarray(pop_layout_files[item])
        pop[item] = I.dot(pop_layout.stack(spatial=("y", "x")))
    
    pop_df = pd.DataFrame(pop, index=clustered_regions_gdf.index)
    
    # Add country information
    pop_df["ct"] = pop_df.index.str[:2]
    
    # Calculate country-level population totals
    country_population = pop_df.total.groupby(pop_df.ct).sum()
    pop_df["fraction"] = pop_df.total / pop_df.ct.map(country_population)
    
    # Calculate load distribution factors based on GDP and population
    load_factors = calculate_load_distribution_factors(
        clustered_regions_gdf, nuts3_shapes, distribution_key
    )
    
    # Add load factors to the dataframe
    # Flatten the multi-index and align with pop_df index
    load_factors_flat = load_factors.reset_index(level=0, drop=True)
    pop_df["load_factor_gdp_pop"] = load_factors_flat.reindex(pop_df.index)
    
    # Calculate alternative load factors based on population only
    pop_df["load_factor_pop_only"] = pop_df.groupby("ct")["total"].transform(
        lambda x: x / x.sum()
    )
    
    # Calculate load factors based on urban population (useful for some applications)
    pop_df["load_factor_urban"] = pop_df.groupby("ct")["urban"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    
    return pop_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_clustered_load_ratio_layouts", clusters=75)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load cutout
    cutout = load_cutout(snakemake.input.cutout)
    
    # Get distribution key from parameters
    distribution_key = snakemake.params.distribution_key
    
    # Prepare population layout files dictionary
    pop_layout_files = {
        "total": snakemake.input.pop_layout_total,
        "urban": snakemake.input.pop_layout_urban, 
        "rural": snakemake.input.pop_layout_rural,
    }
    
    # Build clustered load ratios
    load_ratios = build_clustered_load_ratios(
        clustered_regions_file=snakemake.input.regions_onshore,
        pop_layout_files=pop_layout_files,
        nuts3_shapes_file=snakemake.input.nuts3_shapes,
        cutout=cutout,
        distribution_key=distribution_key,
    )
    
    # Save results
    load_ratios.to_csv(snakemake.output.clustered_load_ratio_layout)
    
    logger.info(f"Clustered load ratio layouts saved to {snakemake.output.clustered_load_ratio_layout}")
