# Standard imports
import os
import sys

# Restarts the script with a fresh interpreter state and forces the number of threads to be used.
# (this shouldn't actually be necessary, but is included for full control in case of a misbehaving environment)
MAX_PARALLEL_WORKERS = min(os.cpu_count(), 48)  # Use up to 48 workers or all available CPUs, whichever is smaller
if "THREAD_CONTROL_INIT" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = f"{MAX_PARALLEL_WORKERS}"
    os.environ["NUMBA_NUM_THREADS"] = f"{MAX_PARALLEL_WORKERS}"
    os.environ["NUMBA_DEFAULT_NUM_THREADS"] = f"{MAX_PARALLEL_WORKERS}"
    os.environ["THREAD_CONTROL_INIT"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from numba import njit, set_num_threads
set_num_threads(MAX_PARALLEL_WORKERS) # For some reason this is necessary to get pykdtree to use the correct number of threads

# Remaining standard imports
import gc
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import re
from pathlib import Path
import contextlib
import zipfile
from io import TextIOWrapper

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm, beta
from scipy.special import gamma, digamma
from pykdtree.kdtree import KDTree

# Astro-specific imports
from astropy.table import Table # Works using v6.1.2, but v7.1.0 seems to try and convert 'null' values to float before using fill_values
from astropy.coordinates import SkyCoord
import astropy.units as u
from gaiaunlimited.selectionfunctions import m10_to_completeness
import galstreams # Also needs astropy==6.1.2 as well as gala==1.9.1

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

# AstroLink imports
from astrolink import AstroLink
from astrolink.io import loadAstroLinkObject, saveAstroLinkObject
from astrolink.visualize import prominenceModel


# === Define script configuration ===
# User-defined paths
GDR3_CATALOGUE_PATH = "/home/_data/Gaia/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"  # Path to raw gdr3 catalogue files
AUXILLARY_CATALOGUES_PATH = "/home/williamoliver_data/gaia_clustering/auxillary_catalogues/"  # Path to auxillary catalogues (e.g. Bailer-Jones GEDR3 distances, Hunt+2024 open clusters)
OUTPUT_PATH = "/home/williamoliver_data/gaia_clustering/"  # Path to output files

# Auto-defined paths
REDUCED_CATALOGUE_PATH = os.path.join(OUTPUT_PATH, "catalogue_files/")  # Path to reduced catalogue numpy files
SUBSAMPLE_PATH = os.path.join(OUTPUT_PATH, "subsample_files/")  # Path to numpy files of subsample from full catalogue
CLUSTERING_PATH = os.path.join(OUTPUT_PATH, "clustering_files/")  # Path to AstroLink output files
FIGURES_PATH = os.path.join(OUTPUT_PATH, "figures_S_gaia_no_floor/")  # Path to figures

# Working memory for k-nearest-neighbour retrieval
WORKING_MEMORY = 200  # GB for max memory usage by k nearest neighbour retrieval 

# Pipeline constants
WITH_PROPER_MOTIONS = True # Whether to use proper motions in the input data space for AstroLink clustering
WITH_RADIAL_VELOCITIES = False # Whether to use radial velocities in the input data space for AstroLink clustering
STOCHASTIC_RUN = False # Whether to sample stochastic values from their uncertainties / posterior
KNN_FOR_SELECTION_FUNCTION = 32 # Number of nearest neighbors for selection function calculations
SURVEY_SF_LOWER_LIMIT = 0.99 # Empirical survey selection function lower limit for subsample stars
RUWE_UPPER_LIMIT = 1.2 # RUWE threshold for subsample stars
HEALPIX_LEVEL = 12 # HEALPix level for on-sky plotting
KNN_FOR_ASTROLINK = 16 # Number of nearest neighbors for AstroLink
SIGMA_THRESHOLDS_FOR_COMPARISONS = np.linspace(2, 10, 81)  # Significance levels from 2 to 10 to be used when comparing to existing cluster catalogues (must be increasing!)
OPTIMAL_SIGMA_THRESHOLD = 3.8 + STOCHASTIC_RUN * np.random.normal(0, 0.1, 1)[0]  # Optimal significance threshold determined from prominence model fitting


# === Reduce GDR3 and Bailer-Jones GEDR3 catalogues to numpy files ===
def prepare_gdr3_catalogue(overwrite=False):
    """
    Reduce raw Gaia catalogue CSV files to grouped numpy arrays.
    """
    # Define column groups for reduction
    column_groups = {
        'source_ids': ['source_id'],
        'galactic_coordinates': ['l', 'b'],
        'equatorial_coordinates': ['ra', 'dec'],
        'proper_motions': ['pmra', 'pmdec'],
        'astrometric_errors': ['ra_error', 'dec_error', 'pmra_error', 'pmdec_error'],
        'astrometric_matched_transits': ['astrometric_matched_transits'],
        'photometry': ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
        'photometric_snr': ['phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error', 'phot_rp_mean_flux_over_error'],
        'ruwe': ['ruwe']
    }

    file_paths = [
        os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}.npy')
        for group_name in column_groups
    ]

    # Skip processing if all merged output files already exist
    all_exist = all(
        os.path.exists(file_path)
        for file_path in file_paths
    )
    if all_exist and not overwrite:
        print("Reduced GDR3 catalogue numpy files already exist at:")
        for i, file_path in enumerate(file_paths):
            if len(file_paths) > 2 and i < len(file_paths) - 2:
                print(f"\t{file_path} ,")
            if len(file_paths) > 1 and i == len(file_paths) - 2:
                print(f"\t{file_path} , and")
            if i == len(file_paths) - 1:
                print(f"\t{file_path} .")
        print("Use overwrite=True to force reprocessing.\n")
        return
    print("Reducing raw Gaia catalogue to numpy files...")

    file_paths = sorted(glob(os.path.join(GDR3_CATALOGUE_PATH, 'GaiaSource_*.csv.gz')))
    print(f"... found {len(file_paths)} source files.")

    # Parallel processing
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = [
            executor.submit(_process_single_gdr3_source_file, index, file_path, column_groups)
            for index, file_path in enumerate(file_paths)
        ]
        for future in futures:
            future.result()  # Propagate any errors

    # Merge all intermediate .npy files by group
    print("... merging temporary numpy files into final arrays and saving them")
    for group_name in column_groups.keys():
        group_files = sorted(glob(os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}_*.npy')))
        arrays = [np.load(f) for f in group_files]
        combined = np.concatenate(arrays).squeeze()

        final_path = os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}.npy')
        np.save(final_path, combined)
        print(f"... saved combined array: {final_path} (shape: {combined.shape})")

        del combined, arrays  # Free memory
        gc.collect() # Force garbage collection

        # Delete intermediates
        for f in group_files:
            os.remove(f)
    print("... reduction complete. All column groups saved as .npy files.\n")

def _process_single_gdr3_source_file(index, file_path, column_groups):
    """Process a single GaiaSource CSV file into group-wise .npy files."""
    # Extract chunk name from filename
    chunk_name = os.path.basename(file_path).replace('GaiaSource_', '').replace('.csv.gz', '')

    # Skip processing if all output files for this chunk already exist
    all_exist = all(
        os.path.exists(os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}_{chunk_name}.npy'))
        for group_name in column_groups
    )
    if all_exist:
        return True

    print(f"... [PROCESS] File {index} ({chunk_name}) — starting in PID {os.getpid():<15}", end='\r')
    
    # Build union of required columns
    all_columns = [col for cols in column_groups.values() for col in cols]

    # Read with astropy
    table = Table.read(file_path, format='ascii.ecsv', include_names=all_columns, fill_values=[("null", "nan")])

    # Convert and save each group
    for group_name, group_cols in column_groups.items():
        columns_data = [np.array(table[col]) for col in group_cols]  # each is 1D array of length n
        array = np.column_stack(columns_data)
        file_path = os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}_{chunk_name}.npy')
        np.save(file_path, array)

    del table, columns_data, array  # Free memory
    gc.collect()  # Force garbage collection

    return True

def prepare_bailerjones_gedr3_distances(overwrite=False):
    """
    Reads the Bailer-Jones et al. 2021 GEDR3 distances dump file, converts 
    columns to numpy arrays. Then re-index Bailer-Jones arrays to match 
    GDR3 source IDs.
    """
    # Define column groups for reduction
    column_groups = {
        'source_ids': ['source_id'],
        'r_med_photogeo': ['r_med_photogeo'],
        'r_lo_high_photogeo': ['r_lo_photogeo', 'r_hi_photogeo']
    }

    file_paths = [
        os.path.join(REDUCED_CATALOGUE_PATH, f'bailerjones_{group_name}.npy')
        for group_name in column_groups
        if group_name != 'source_ids'  # final *_source_ids.npy file is from the GDR3 catalogue
    ]

    # Skip processing if all merged output files already exist
    all_exist = all(
        os.path.exists(file_path)
        for file_path in file_paths
    )
    if all_exist and not overwrite:
        print("Reduced Bailer-Jones et al. 2021 distance numpy files already exist at:")
        print(f"\t{file_paths[0]} and")
        print(f"\t{file_paths[1]} .")
        print("Use overwrite=True to force reprocessing.\n")
        return
    print("Reducing the Bailer-Jones et al. 2021 distance dump file to numpy arrays...")

    chunksize = 10**6  # Adjust based on your memory
    all_columns = [col for cols in column_groups.values() for col in cols]
    bailerjones_gedr3_distances_file = os.path.join(AUXILLARY_CATALOGUES_PATH, 'BailerJones2021/gedr3dist.dump.gz')
    for i, chunk in enumerate(pd.read_csv(bailerjones_gedr3_distances_file, compression="gzip", chunksize=chunksize, usecols=all_columns)):
        print(f"... processing data in chunks, {i+1} of {1467744818//chunksize + 1}", end='\r')
        # Skip processing if all output files for this chunk already exist
        all_exist = all(
            os.path.exists(os.path.join(REDUCED_CATALOGUE_PATH, f'bailerjones_{group_name}_{i}.npy'))
            for group_name in column_groups
        )
        if all_exist:
            continue
        
        # Convert and save each group
        for group_name, group_cols in column_groups.items():
            columns_data = [np.array(chunk[col]) for col in group_cols]  # each is 1D array of length n
            array = np.column_stack(columns_data).squeeze()
            file_path = os.path.join(REDUCED_CATALOGUE_PATH, f'bailerjones_{group_name}_{i}.npy')
            np.save(file_path, array)
        
        del chunk, columns_data, array, file_path  # Free memory
        gc.collect()

    # Merge all intermediate .npy files by group
    print("... merging temporary numpy files into final arrays (indexed with respect to the gdr3 catalogue)")
    for i, group_name in enumerate(column_groups.keys()):
        group_files = sorted(glob(os.path.join(REDUCED_CATALOGUE_PATH, f'bailerjones_{group_name}_*.npy')))
        arrays = [np.load(f) for f in group_files]
        combined = np.concatenate(arrays)

        if i == 0:  # source_id is the first group, so we can use it to re-index
            # Load GDR3 source IDs and Bailer-Jones source IDs
            print("... loading GDR3 and Bailer-Jones source IDs")
            gdr3_source_ids = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_source_ids.npy"))  # (n,)
            n = gdr3_source_ids.size

            # Compute index lookup for Bailer-Jones source IDs in GDR3
            print("... computing index lookup for Bailer-Jones source IDs in GDR3")
            indices = np.searchsorted(gdr3_source_ids, combined) # Assumes gdr3_source_ids is sorted

            del gdr3_source_ids  # Free memory
            gc.collect()  # Force garbage collection
        else:
            # Re-index the combined array to match GDR3 source IDs
            if combined.ndim > 1:
                combined_reindexed = np.full((n, combined.shape[1]), np.nan)  # Initialize with NaNs
            else:
                combined_reindexed = np.full(n, np.nan)
            combined_reindexed[indices] = combined

            # Save the re-indexed array
            final_path = os.path.join(REDUCED_CATALOGUE_PATH, f'bailerjones_{group_name}.npy')
            np.save(final_path, combined_reindexed)
            print(f"... saved re-indexed combined array: {final_path} (shape: {combined_reindexed.shape})")

            del combined_reindexed  # Free memory
            gc.collect()  # Force garbage collection

        del combined, arrays  # Free memory
        gc.collect() # Force garbage collection

        # Delete intermediates
        for f in group_files:
            os.remove(f)
    print("... reduction complete. All column groups saved as .npy files.\n")
    

# === Calculate empirical selection function ===
def calculate_empirical_survey_selection_function(overwrite=False):
    """
    Compute the empirical survey selection function using a kNN-based M10 metric
    and save each as a .npy files aligned with the G-band photometry array.
    """
    # Check if selection function already exists
    file_path_m10_stars = os.path.join(REDUCED_CATALOGUE_PATH, "m10_stars.npy")
    file_path_sf = os.path.join(REDUCED_CATALOGUE_PATH, "empirical_survey_selection_function.npy")
    file_path_m10_healpix = os.path.join(REDUCED_CATALOGUE_PATH, "m10_healpix.npy")
    if os.path.exists(file_path_m10_stars) and os.path.exists(file_path_sf) and os.path.exists(file_path_m10_healpix) and not overwrite:
        print(f"Empirical selection function and m10 values for the centre of HEALPix pixels already exist at:")
        print(f"\t{file_path_sf} and")
        print(f"\t{file_path_m10_healpix} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating empirical survey selection function...")

    # Load required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))  # shape (n, 2)
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))[:, 0]           # shape (n,)
    astrometric_matched_transits = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_astrometric_matched_transits.npy")) # shape (n,)

    # Identify stars with valid G magnitude and also stars with less than 11 astrometric matched transits
    print("... identifying valid G-band magnitudes and astrometric matched transits")
    valid_gmag = np.isfinite(G_band_magnitudes)
    valid_for_kNN = np.where(valid_gmag & (astrometric_matched_transits < 11))[0]  # Indices of stars with valid G-band magnitudes and <11 astrometric matched transits
    valid_for_kNN = valid_for_kNN.astype(np.int32)
    del astrometric_matched_transits  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians
    print("... converting galactic coordinates to radians")
    l_rad, b_rad = np.deg2rad(galactic_coordinates).T
    del galactic_coordinates  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in radians to unit 3D Cartesian coordinates
    print("... converting galactic coordinates to unit 3D Cartesian coordinates")
    cos_b = np.cos(b_rad)
    xyz_stars = np.column_stack([
        np.cos(l_rad) * cos_b,  # x
        np.sin(l_rad) * cos_b,  # y
        np.sin(b_rad)           # z
    ]) # Positions on the sky (in 3D Cartesian) of all stars
    del l_rad, b_rad, cos_b  # Free memory
    gc.collect()  # Force garbage collection

    # Build KDTree with only those stars with valid G-band magnitudes and with less than 11 astrometric matched transits
    print("... building kNN tree from the unit 3D Cartesian coordinates of valid stars")
    n = xyz_stars.shape[0]
    m10_stars = np.full_like(G_band_magnitudes, np.nan)  # Initialize m10 values for stars
    tree = KDTree(xyz_stars[valid_for_kNN])

    # Batching for memory efficiency
    chunk_n_rows = min(int(WORKING_MEMORY * (2**30) // (16 * KNN_FOR_SELECTION_FUNCTION)), n)

    # Compute m10 for each star as median G of neighbors with <11 transits
    for start in range(0, n, chunk_n_rows):
        end = min(start + chunk_n_rows, n)

        print(f"... computing m10 values for each star -- batch {start // chunk_n_rows + 1} of {n // chunk_n_rows + 1}   ", end='\r')
        # k-nearest neighbours query
        sqr_dists, idx = tree.query(
            xyz_stars[start:end],
            k=KNN_FOR_SELECTION_FUNCTION,
            sqr_dists=True
        )
        del sqr_dists  # Free memory
        gc.collect()  # Force garbage collection

        # Median G-band magnitude of neighbors
        m10_stars[start:end] = np.median(G_band_magnitudes[valid_for_kNN[idx]], axis=1)
        del idx  # Free memory
        gc.collect()  # Force garbage collection

    # Save m10 values for stars
    print(f"... saving m10 values for stars to {file_path_m10_stars} (shape: {m10_stars.shape})")
    np.save(file_path_m10_stars, m10_stars)
    
    # Compute completeness using m10_to_completeness (only for valid G-band magnitudes)
    print("... calculating empirical survey selection function using m10_to_completeness")
    selection_function = np.full_like(G_band_magnitudes, np.nan)
    selection_function[valid_gmag] = m10_to_completeness(
        G_band_magnitudes[valid_gmag],
        m10_stars[valid_gmag]
    )
    del valid_gmag, m10_stars, xyz_stars  # Free memory
    gc.collect()  # Force garbage collection
    print(f"... empirical survey selection function range: {np.nanmin(selection_function):.3f} -- {np.nanmax(selection_function):.3f}")

    # Save the selection function
    print(f"... saving empirical survey selection function to {file_path_sf} (valid: {np.isfinite(selection_function).sum()} stars)\n")
    np.save(file_path_sf, selection_function)
    del selection_function  # Free memory
    gc.collect()  # Force garbage collection

    # Also calculate m10 values at the centre of each HEALPix pixel for plotting
    print("Calculating m10 values for HEALPix pixels...")
    nside = 2**HEALPIX_LEVEL
    npix = hp.nside2npix(nside)

    # Convert (theta, phi) in degrees to unit 3D Cartesian coordinates
    print("... converting HEALPix pixel centres to unit 3D Cartesian coordinates")
    phi_rad, theta_rad = hp.pix2ang(nside, np.arange(npix), nest=True) # pix2ang returns arrays in phi, theta order
    phi_rad = np.pi / 2 - phi_rad  # Convert phi from [0, pi] to [-pi/2, pi/2]
    cos_phi = np.cos(phi_rad)
    xyz_healpix = np.column_stack([
        np.cos(theta_rad) * cos_phi,  # x
        np.sin(theta_rad) * cos_phi,  # y
        np.sin(phi_rad)               # z
    ]) # Positions on the sky (in 3D Cartesian) of HEALPix pixels
    del theta_rad, phi_rad, cos_phi  # Free memory
    gc.collect()  # Force garbage collection

    # Update chunking for HEALPix
    print("... updating chunk size for HEALPix pixels")
    chunk_n_rows = min(int(WORKING_MEMORY * (2**30) // (16 * KNN_FOR_SELECTION_FUNCTION)), npix)

    # Initialize m10 array for HEALPix pixels
    print("... initializing m10 array for HEALPix pixels")
    m10_healpix = np.empty(npix)

    # Compute m10 for each HEALPix pixel as median G of neighbors with <11 transits
    for start in range(0, n, chunk_n_rows):
        end = min(start + chunk_n_rows, n)

        print(f"... computing m10 values for HEALPix pixels -- batch {start // chunk_n_rows + 1} of {n // chunk_n_rows + 1}     ", end='\r')
        # k-nearest neighbours query
        _, idx = tree.query(
            xyz_healpix[start:end],
            k=KNN_FOR_SELECTION_FUNCTION,
            sqr_dists=True
        )

        # Median G-band magnitude of neighbors
        m10_healpix[start:end] = np.median(G_band_magnitudes[valid_for_kNN[idx]], axis=1)

        # Delete temporary variables to free memory
        del _, idx
        gc.collect()
    
    del tree, xyz_healpix, valid_for_kNN  # Free memory
    gc.collect()  # Force garbage collection

    # Save m10 values for HEALPix pixels
    print(f"... saving m10 values for HEALPix pixels to {file_path_m10_healpix} (shape: {m10_healpix.shape})\n")
    np.save(file_path_m10_healpix, m10_healpix)
    del m10_healpix  # Free memory
    gc.collect()  # Force garbage collection

def plot_limiting_g_band_magnitude_on_sky(overwrite=False):
    """
    Plot the limiting G-band magnitude across the sky using HEALPix.
    """
    # Check if plot already exists
    file_m10_path = os.path.join(FIGURES_PATH, "m10_map.png")
    file_limiting_g_mag_path = os.path.join(FIGURES_PATH, "limiting_g_mag.png")
    if os.path.exists(file_m10_path) and os.path.exists(file_limiting_g_mag_path) and not overwrite:
        print(f"Plots of m10 map and limiting G-band magnitude already exist at:")
        print(f"\t{file_m10_path} and")
        print(f"\t{file_limiting_g_mag_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting M10 map across the sky...")

    # Load m10 values for HEALPix pixels
    m10 = np.load(f"{REDUCED_CATALOGUE_PATH}/m10_healpix.npy")  # (npix,)

    # Create a Mollweide projection plot of the limiting G-band magnitude
    plt.figure(figsize=(12, 6))
    projview(
        m10,
        coord=["G"],
        nest=True,
        unit=r"$M_{10}$",
        cb_orientation="horizontal",
        min=20,
        max=np.ceil(10**2 * m10.max()) / 10**2,
        cmap="magma_r",
        projection_type="mollweide",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_m10_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    
    print(f"... saved mollview plot to {file_m10_path}\n")


    print("Plotting limiting G-band magnitude across the sky...")
    # Taken from the source code of gaiaunlimited.selectionfunctions.m10_to_completeness...
    # These are the best-fit value of the free parameters we optimised in our model:
    ax, bx, cx, ay, by, cy, az, bz, cz, lim = dict(
        ax=0.9848761394197864,
        bx=0.6473155510230146,
        cx=0.6929084598209412,
        ay=-0.003935382139847386,
        by=0.2230529402297744,
        cy=-0.09331877468160235,
        az=0.006144107896473064,
        bz=0.03681705933744438,
        cz=0.35140564525722895,
        lim=20.519369625540833,
    ).values()

    predictedG0 = ax * m10 + bx
    predictedG0[m10 > lim] = cx * m10[m10 > lim] + (ax - cx) * lim + bx
    #
    predictedInvslope = ay * m10 + by
    predictedInvslope[m10 > lim] = cy * m10[m10 > lim] + (ay - cy) * lim + by
    #
    predictedShape = az * m10 + bz
    predictedShape[m10 > lim] = cz * m10[m10 > lim] + (az - cz) * lim + bz

    # Calculate the inverse of the selection function given the m10 values
    limiting_g_band_magnitude = predictedG0 + predictedInvslope * np.arctanh(2 * (1 - SURVEY_SF_LOWER_LIMIT) ** (1 / predictedShape) - 1)

    # Create a Mollweide projection plot of the limiting G-band magnitude
    plt.figure(figsize=(12, 6))
    projview(
        limiting_g_band_magnitude,
        coord=["G"],
        nest=True,
        unit=r"Limiting $G$-band magnitude",
        cb_orientation="horizontal",
        min=20,
        max=np.ceil(10**2 * limiting_g_band_magnitude.max()) / 10**2,
        cmap="magma_r",
        projection_type="mollweide",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_limiting_g_mag_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    
    print(f"... saved mollview plot to {file_limiting_g_mag_path}.\n")


# === Construct subsample and subsample selection function ===
def construct_subsample_from_full_catalogue(overwrite=False):
    """
    Create a boolean subsample mask where the empirical survey selection function S_Gaia > SURVEY_SF_LOWER_LIMIT.
    """
    # Check if subsample mask already exists
    mask_path = os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy")
    if os.path.exists(mask_path) and not overwrite:
        print(f"Subsample mask already exists at:\n\t{mask_path} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Constructing subsample from full catalogue...")

    # Load selection function and galactic coordinates
    print("... loading required arrays from reduced catalogue")
    selection_function = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "empirical_survey_selection_function.npy"))  # (n,)
    galactic_coords = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))  # (n, 2) in degrees
    r_med_photogeo = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_med_photogeo.npy"))  # (n,)
    proper_motions = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_proper_motions.npy"))  # (n, 2) in mas/yr
    ruwe = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_ruwe.npy"))  # (n,)
    g_mag = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))[:, 0]  # (n,)

    # EXPERIMENTAL: Load contracted uncertainties to define threshold for removing more stars with high uncertainties
    #sigma_pos = np.load(os.path.join(OUTPUT_PATH, "subsample_files/contracted_position_uncertainties.npy"))
    #sigma_vel = np.load(os.path.join(OUTPUT_PATH, "subsample_files/contracted_velocity_uncertainties.npy"))
    #sigma_pos /= np.median(sigma_pos)
    #sigma_vel /= np.median(sigma_vel)
    #total_uncertainty_squared = sigma_pos**2 + sigma_vel**2
    #threshold = np.percentile(total_uncertainty_squared, 100 * 6/11)

    # Create boolean mask for S_Gaia > threshold, ruwe < threshold, and valid astrometric data
    print("... creating subsample mask with selection function and data quality cuts")
    subsample_mask = selection_function > SURVEY_SF_LOWER_LIMIT
    subsample_mask &= ruwe < RUWE_UPPER_LIMIT
    subsample_mask &= np.isfinite(r_med_photogeo)
    subsample_mask &= np.isfinite(proper_motions).all(axis=1)
    subsample_mask &= np.isfinite(g_mag)  # Valid G-band magnitude

    # EXPERIMENTAL: Further subsample to remove stars with high uncertainties
    #subsample_where = np.where(subsample_mask)[0][total_uncertainty_squared > threshold]
    #subsample_mask[subsample_where] = False

    # Save mask
    np.save(mask_path, subsample_mask)
    print(f"... saved subsample mask to {mask_path} (selected {subsample_mask.sum()} stars).\n")

def calculate_subsample_selection_function(overwrite=False):
    """
    Calculate the subsample selection function using kNN-based metric.
    """
    # Check if total selection function already exists
    file_subsample_sf_stars = os.path.join(SUBSAMPLE_PATH, "subsample_selection_function.npy")
    if os.path.exists(file_subsample_sf_stars) and not overwrite:
        print(f"Subsample selection function already exists at:\n\t{file_subsample_sf_stars} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating subsample selection function...")

    # Load required arrays
    print("... loading required arrays")
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))  # shape (n, 2)
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))[:, 0]  # shape (n,)
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # shape (n,)

    # Identify stars with valid G magnitude
    print("... identifying valid G-band magnitudes")
    valid_gmag = np.where(np.isfinite(G_band_magnitudes))[0]

    # Convert (l, b) in degrees to radians
    print("... converting galactic coordinates to radians")
    l_rad, b_rad = np.deg2rad(galactic_coordinates[valid_gmag]).T
    del galactic_coordinates  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in radians to unit 3D Cartesian coordinates
    print("... converting galactic coordinates to unit 3D Cartesian coordinates")
    cos_l = np.cos(l_rad)
    sin_l = np.sin(l_rad)
    cos_b = np.cos(b_rad)
    sin_b = np.sin(b_rad)
    xyz_stars = np.column_stack([
        cos_l * cos_b,  # x
        sin_l * cos_b,  # y
        sin_b           # z
    ]) # Positions on the sky (in 3D Cartesian) of all stars
    del l_rad, b_rad, cos_l, sin_l, cos_b, sin_b  # Free memory
    gc.collect()  # Force garbage collection

    # Construct weighted composite space for kNN (balances HEALPix level 5 area with 0.2 mags)
    print("... constructing weighted composite space for kNN")
    nside = 2**5
    npix = hp.nside2npix(nside)
    x_scale = 4 * np.pi / npix # Scale factor for Cartesian coordinates
    G_scale = 0.2  # Scale factor for G-band magnitude
    comp_stars = np.concatenate([
        xyz_stars,  # 3D Cartesian coordinates
        G_band_magnitudes[valid_gmag, np.newaxis] * x_scale / G_scale  # Rescaled G-band magnitude as a 4th dimension
    ], axis=1)  # shape (n_valid, 4)
    del xyz_stars  # Free memory
    gc.collect()  # Force garbage collection

    # Build KDTree with only those stars with valid G-band magnitudes
    print("... building kNN tree from the 4D composite coordinates of valid stars")
    n = comp_stars.shape[0]
    subsample_sf = np.full_like(G_band_magnitudes, np.nan)  # Initialize m10 values for stars
    tree = KDTree(comp_stars)
    del G_band_magnitudes  # Free memory
    gc.collect()  # Force garbage collection

    # Batching for memory efficiency
    chunk_n_rows = min(int(WORKING_MEMORY * (2**30) // (16 * KNN_FOR_SELECTION_FUNCTION)), n)

    # Compute subsample selection function for each star as fraction of neighbourhood in subsample
    for start in range(0, n, chunk_n_rows):
        end = min(start + chunk_n_rows, n)

        print(f"... computing subsample selection function for each star -- batch {start // chunk_n_rows + 1} of {n // chunk_n_rows + 1}   ", end='\r')
        # k-nearest neighbours query
        _, idx = tree.query(
            comp_stars[start:end],
            k=KNN_FOR_SELECTION_FUNCTION,
            sqr_dists=True
        )

        # Fraction of neighbours in subsample
        subsample_sf[valid_gmag[start:end]] = subsample_mask[valid_gmag[idx]].sum(axis=1) / KNN_FOR_SELECTION_FUNCTION

        # Delete temporary variables to free memory
        del _, idx
        gc.collect()
    print(f"... subsample selection function range: {np.nanmin(subsample_sf):.3f} -- {np.nanmax(subsample_sf):.3f}                            ")

    # Save subsample selection function for stars
    print(f"... saving subsample selection function for stars to {file_subsample_sf_stars} (shape: {subsample_sf.shape}).\n")
    np.save(file_subsample_sf_stars, subsample_sf)
    del subsample_sf, comp_stars, tree  # Free memory
    gc.collect()  # Force garbage collection


# === Calculate total selection function for subsample ===
def calculate_total_selection_function_for_subsample(overwrite=False):
    """
    Calculate the total selection function for the subsample.
    """
    # Check if arrays already exists
    file_nsub = os.path.join(SUBSAMPLE_PATH, "total_selection_function_nsub.npy")
    file_nmw = os.path.join(SUBSAMPLE_PATH, "total_selection_function_nmw.npy")
    file_total_sf_mean = os.path.join(SUBSAMPLE_PATH, "total_selection_function_mean.npy")
    file_total_sf_var = os.path.join(SUBSAMPLE_PATH, "total_selection_function_var.npy")
    file_total_sf_mean_healpix = os.path.join(SUBSAMPLE_PATH, "total_selection_function_mean_healpix.npy")
    file_total_sf_var_healpix = os.path.join(SUBSAMPLE_PATH, "total_selection_function_var_healpix.npy")

    # Skip processing if all output files already exist
    all_exist = (os.path.exists(file_nsub) and
                os.path.exists(file_nmw) and
                os.path.exists(file_total_sf_mean) and
                os.path.exists(file_total_sf_var) and
                os.path.exists(file_total_sf_mean_healpix) and
                os.path.exists(file_total_sf_var_healpix))
    if all_exist and not overwrite:
        print(f"Total selection function arrays already exist at:")
        print(f"\t{file_total_sf_mean} ,")
        print(f"\t{file_total_sf_var} ,")
        print(f"\t{file_nsub} ,")
        print(f"\t{file_nmw} ,")
        print(f"\t{file_total_sf_mean_healpix} , and")
        print(f"\t{file_total_sf_var_healpix} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating total selection function for the subsample...")

    # Load the required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))[:, 0]
    survey_sf = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "empirical_survey_selection_function.npy"))
    subsample_sf = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_selection_function.npy"))

    # Identify stars with valid G magnitude
    print("... identifying valid G-band magnitudes")
    valid_gmag = np.isfinite(G_band_magnitudes)
    del G_band_magnitudes  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate the inverse of the empirical survey selection function for the subsample
    inverse_survey_sf = 1 / survey_sf[valid_gmag]#np.sqrt(survey_sf[valid_gmag]**2 + 1 / KNN_FOR_SELECTION_FUNCTION**2)  # Soft floor to avoid diverging values
    del survey_sf  # Free memory
    gc.collect()  # Force garbage collection

    # Make subsample selection function relative to valid G-band magnitudes
    subsample_sf = subsample_sf[valid_gmag]

    # Convert (l, b) in degrees to radians
    print("... converting galactic coordinates to radians")
    l_rad, b_rad = np.deg2rad(galactic_coordinates[valid_gmag]).T
    del galactic_coordinates  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in radians to unit 3D Cartesian coordinates
    print("... converting galactic coordinates to unit 3D Cartesian coordinates")
    cos_l = np.cos(l_rad)
    sin_l = np.sin(l_rad)
    cos_b = np.cos(b_rad)
    sin_b = np.sin(b_rad)
    xyz_stars = np.column_stack([
        cos_l * cos_b,  # x
        sin_l * cos_b,  # y
        sin_b           # z
    ]) # Positions on the sky (in 3D Cartesian) of all stars
    del l_rad, b_rad, cos_l, sin_l, cos_b, sin_b  # Free memory
    gc.collect()  # Force garbage collection

    # Build KDTree with only those stars with valid G-band magnitudes
    print("... building kNN tree from the unit 3D Cartesian coordinates of valid stars")
    n = subsample_sf.size # Number of stars in the subsample
    tree = KDTree(xyz_stars) # Build KDTree with all stars with valid G-band magnitudes

    # Batching for memory efficiency
    chunk_n_rows = min(int(WORKING_MEMORY * (2**30) // (16 * KNN_FOR_SELECTION_FUNCTION)), n)

    # Initialize total selection function array for stars in the subsample
    print("... initializing total selection function arrays for stars in the subsample")
    nsub = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    nmw = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    total_sf_mean = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    total_sf_var = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    valid_gmag = np.where(valid_gmag)[0]  # Indices of stars with valid G-band magnitudes

    # Compute total selection function for each star in the subsample
    for start in range(0, n, chunk_n_rows):
        end = min(start + chunk_n_rows, n)

        print(f"... computing total selection function for each star in subsample -- batch {start // chunk_n_rows + 1} of {n // chunk_n_rows + 1}   ", end='\r')
        # k-nearest neighbours query
        sqr_dists, idx = tree.query(
            xyz_stars[start:end],
            k=KNN_FOR_SELECTION_FUNCTION,
            sqr_dists=True
        )
        del sqr_dists  # Free memory
        gc.collect()  # Force garbage collection

        # Total selection function is the posterior distribution Beta(n_sub + 1, n_mw - n_sub + 1)
        nsub_batch = subsample_sf[idx].sum(axis=1)
        nmw_batch = inverse_survey_sf[idx].sum(axis=1)
        valid_slice = valid_gmag[start:end]
        nsub[valid_slice] = nsub_batch
        nmw[valid_slice] = nmw_batch
        total_sf_mean[valid_slice] = (nsub_batch + 1) / (nmw_batch + 2)  # Mean of selection function for stars in subsample
        total_sf_var[valid_slice] = (nsub_batch + 1) * (nmw_batch - nsub_batch + 1) / ((nmw_batch + 2)**2 * (nmw_batch + 3))  # Variance of selection function for stars in subsample

        # Delete temporary variables to free memory
        del idx, nsub_batch, nmw_batch
        gc.collect()
    print(f"... range of expected number of neighbours in subsample: {np.nanmin(nsub):.3f} -- {np.nanmax(nsub):.3f}                   ")
    print(f"... range of expected number of neighbours in Milky Way: {np.nanmin(nmw):.3f} -- {np.nanmax(nmw):.3f}")
    print(f"... range of total selection function mean:              {np.nanmin(total_sf_mean):.3f} -- {np.nanmax(total_sf_mean):.3f}")
    print(f"... range of total selection function variance:          {np.nanmin(total_sf_var):.3f} -- {np.nanmax(total_sf_var):.3f}")

    # Save total selection function arrays for stars
    print(f"... saving total selection function arrays for stars to:")
    print(f"\t{file_nsub} ,")
    print(f"\t{file_nmw} ,")
    print(f"\t{file_total_sf_mean} , and")
    print(f"\t{file_total_sf_var} .\n")
    np.save(file_nsub, nsub)
    np.save(file_nmw, nmw)
    np.save(file_total_sf_mean, total_sf_mean)
    np.save(file_total_sf_var, total_sf_var)

    del nsub, nmw, total_sf_mean, total_sf_var, xyz_stars  # Free memory
    gc.collect()  # Force garbage collection

    # Also calculate the total selection function values at the centre of each HEALPix pixel for plotting
    print("Calculating total selection function for HEALPix pixels...")
    nside = 2**HEALPIX_LEVEL
    npix = hp.nside2npix(nside)

    # Convert (theta, phi) in degrees to unit 3D Cartesian coordinates
    print("... converting HEALpix pixel centres to unit 3D Cartesian coordinates")
    phi_rad, theta_rad = hp.pix2ang(nside, np.arange(npix), nest=True) # pix2ang returns arrays in phi, theta order
    phi_rad = np.pi / 2 - phi_rad  # Convert phi from [0, pi] to [-pi/2, pi/2]
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    xyz_healpix = np.column_stack([
        cos_theta * cos_phi,  # x
        sin_theta * cos_phi,  # y
        sin_phi           # z
    ]) # Positions on the sky (in 3D Cartesian) of HEALPix pixels
    del theta_rad, phi_rad, cos_theta, sin_theta, cos_phi, sin_phi  # Free memory
    gc.collect()  # Force garbage collection

    # Update chunking for HEALPix
    print("... updating chunk size for HEALPix pixels")
    chunk_n_rows = min(int(WORKING_MEMORY * (2**30) // (16 * KNN_FOR_SELECTION_FUNCTION)), npix)

    # Initialize arrays for HEALPix pixels
    print("... initializing total selection function arrays for HEALPix pixels")
    total_sf_mean_healpix = np.empty(npix)
    total_sf_var_healpix = np.empty(npix)

    # Compute m10 for each HEALPix pixel as median G of neighbors with <11 transits
    for start in range(0, npix, chunk_n_rows):
        end = min(start + chunk_n_rows, npix)

        print(f"... computing total selection function for HEALPix pixels -- batch {start // chunk_n_rows + 1} of {npix // chunk_n_rows + 1}   ", end='\r')
        # k-nearest neighbours query
        _, idx = tree.query(
            xyz_healpix[start:end],
            k=KNN_FOR_SELECTION_FUNCTION,
            sqr_dists=True
        )

        # Total selection function is the posterior distribution Beta(n_sub + 1, n_mw - n_sub + 1)
        nsub_healpix_batch = subsample_sf[idx].sum(axis=1)
        nmw_healpix_batch = inverse_survey_sf[idx].sum(axis=1)
        total_sf_mean_healpix[start:end] = (nsub_healpix_batch + 1) / (nmw_healpix_batch + 2)  # Mean of selection function for HEALPix pixels
        total_sf_var_healpix[start:end] = (nsub_healpix_batch + 1) * (nmw_healpix_batch - nsub_healpix_batch + 1) / ((nmw_healpix_batch + 2)**2 * (nmw_healpix_batch + 3))  # Variance of selection function for HEALPix pixels

        # Delete temporary variables to free memory
        del _, idx, nsub_healpix_batch, nmw_healpix_batch
        gc.collect()
    
    del tree, xyz_healpix, valid_gmag, subsample_sf, inverse_survey_sf  # Free memory
    gc.collect()  # Force garbage collection

    # Save total selection function arrays for healpix pixels
    print(f"... saving total selection function arrays for HEALPix pixels to:                ")
    print(f"\t{file_total_sf_mean_healpix} , and")
    print(f"\t{file_total_sf_var_healpix} .\n")
    np.save(file_total_sf_mean_healpix, total_sf_mean_healpix)
    np.save(file_total_sf_var_healpix, total_sf_var_healpix)
    del total_sf_mean_healpix, total_sf_var_healpix  # Free memory
    gc.collect()  # Force garbage collection

def plot_total_selection_function_for_subsample(overwrite=False):
    """
    Plot the mean and standard error of the total selection function across the sky using HEALPix.
    """
    # Check if plots already exist
    file_total_sf_mean_path = os.path.join(FIGURES_PATH, "total_selection_function_mean.png")
    file_total_sf_se_path = os.path.join(FIGURES_PATH, "total_selection_function_stderr.png")
    file_total_sf_se_over_mean_path = os.path.join(FIGURES_PATH, "total_selection_function_stderr_over_mean.png")
    if os.path.exists(file_total_sf_mean_path) and os.path.exists(file_total_sf_se_path) and not overwrite:
        print(f"Total selection function mean and standard error plots already exist at:")
        print(f"\t{file_total_sf_mean_path} ,")
        print(f"\t{file_total_sf_se_path} , and")
        print(f"\t{file_total_sf_se_over_mean_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting total selection function on the sky...")

    # Load total selection function data
    total_sf_mean = np.load(os.path.join(SUBSAMPLE_PATH, "total_selection_function_mean_healpix.npy"))  # (npix,)
    total_sf_var = np.load(os.path.join(SUBSAMPLE_PATH, "total_selection_function_var_healpix.npy"))  # (npix,)

    # Create a Mollweide projection plot of the total selection function mean
    plt.figure(figsize=(12, 6))
    projview(
        total_sf_mean,
        coord=["G"],
        nest=True,
        unit=r"Total selection function mean, $\mathbb{E}[S_{\mathrm{total}}]$",
        cb_orientation="horizontal",
        min=0,
        max=1,
        cmap="magma",
        projection_type="mollweide",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_total_sf_mean_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory

    print(f"... saved mollview plot to {file_total_sf_mean_path}")

    # Compute standard error (sqrt of variance)
    total_sf_se = np.sqrt(total_sf_var)
    del total_sf_var  # Free memory
    gc.collect()  # Force garbage collection

    lower_limit_se_scale = 10**np.floor(np.log10(total_sf_se.min()))
    lower_limit_se = lower_limit_se_scale * np.floor(total_sf_se.min() / lower_limit_se_scale)
    upper_limit_se_scale = 10**np.ceil(np.log10(total_sf_se.max()))
    upper_limit_se = upper_limit_se_scale * np.ceil(total_sf_se.max() / upper_limit_se_scale)

    # Create a Mollweide projection plot of the total selection function standard error
    plt.figure(figsize=(12, 6))
    projview(
        total_sf_se,
        coord=["G"],
        nest=True,
        unit=r"Total selection function standard error, $\sqrt{\mathrm{Var}[S_{\mathrm{total}}]}$",
        cb_orientation="horizontal",
        min=lower_limit_se,
        max=upper_limit_se,
        cmap="magma",
        projection_type="mollweide",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_total_sf_se_path, dpi=300)
    plt.close()
    gc.collect()

    print(f"... saved mollview plot to {file_total_sf_se_path}")

    # Compute standard error over mean
    stderr_over_mean = total_sf_se / total_sf_mean
    del total_sf_se, total_sf_mean  # Free memory
    gc.collect()  # Force garbage collection

    lower_limit_se_over_mean_scale = 10**np.floor(np.log10(stderr_over_mean.min()))
    lower_limit_se_over_mean = lower_limit_se_over_mean_scale * np.floor(stderr_over_mean.min() / lower_limit_se_over_mean_scale)

    # Create a Mollweide projection plot of the total selection function standard error over mean
    plt.figure(figsize=(12, 6))
    projview(
        stderr_over_mean,
        coord=["G"],
        nest=True,
        unit=r"Total selection function fractional error, $\sqrt{\mathrm{Var}[S_{\mathrm{total}}]} / \mathbb{E}[S_{\mathrm{total}}]$",
        cb_orientation="horizontal",
        min=lower_limit_se_over_mean,
        max=1,
        cmap="magma",
        projection_type="mollweide",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_total_sf_se_over_mean_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory

    print(f"... saved mollview plot to {file_total_sf_se_over_mean_path}.\n")


# === Construct input data to be passed to AstroLink ===
def calculate_distance_contraction_for_subsample(overwrite=False):
    """
    Calculate the distance contraction for the subsample.
    """
    # Check if the distance contraction already exists
    file_path_r_half = os.path.join(SUBSAMPLE_PATH, "contracted_r_half.npy")
    file_path_fr = os.path.join(SUBSAMPLE_PATH, "contracted_distance.npy")
    file_path_delta_fr = os.path.join(SUBSAMPLE_PATH, "contracted_distance_error.npy")

    all_exist = all(os.path.exists(p) for p in [file_path_r_half, file_path_fr, file_path_delta_fr])
    if all_exist and not overwrite:
        print(f"Distance contraction arrays already exist at:")
        print(f"\t{file_path_r_half} ,")
        print(f"\t{file_path_fr} , and")
        print(f"\t{file_path_delta_fr} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating distance contraction and its error for subsample...")

    # Load required arrays
    print("... loading required arrays")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (n,)
    r = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_med_photogeo.npy"))[subsample_mask]  # (n,) in pc
    lo, high = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_lo_high_photogeo.npy"))[subsample_mask].T  # each (n,) in pc
    ra, dec = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_equatorial_coordinates.npy"))[subsample_mask].T  # each (n,) in degrees
    dra, ddec = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_astrometric_errors.npy"))[subsample_mask, :2].T  # each (n,) in degrees
    total_sf_mean = np.load(os.path.join(SUBSAMPLE_PATH, "total_selection_function_mean.npy"))[subsample_mask]  # (N,)
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection
    
    # Calculate variances in spherical coordinates
    drSqr = (high - lo)**2 / 4
    variances = np.column_stack([
        drSqr,  # Variance in LOS
        (np.cos(np.deg2rad(dec)) * np.deg2rad(dra))**2,  # Variance in RA
        np.deg2rad(ddec)**2 # Variance in Dec
    ])
    zero_variances = drSqr == 0
    variances[zero_variances, 0] = variances[zero_variances, 1:].sum(axis=1) / 2  # Prevent zero variance in distance

    # Weights for each star in the average
    weights = 1 / total_sf_mean
    del lo, high, drSqr, ra, dec, dra, ddec, total_sf_mean  # Free memory
    gc.collect()  # Force garbage collection

    # Define loss wrapper for minimization so that the current r_half and loss can be printed
    def loss_wrapper(r_half, variances, r, weights):
        loss = weighted_average_sym_kl_contracted(r_half, variances, r, weights)
        print("\t... r_{1/2}:" + f"{r_half:10.4f} | loss: {loss:10.6f}")
        return loss

    # Fit model using a grid search for r_{1/2}
    print("... fitting r_{1/2} to get globally isotropic spatial uncertainties")
    bounds = (1, 1000)  # Initial guess for r_{1/2} in pc
    result = minimize_scalar(
        lambda r_half: loss_wrapper(r_half, variances, r, weights),
        bounds=bounds,
        method='bounded',
        options={'xatol': 1.0}      # stop when r_half is within 1 pc
    )
    r_half = result.x  # Best fit characteristic scale r_{1/2} in pc
    print("... best fit r_{1/2} = " + f"{r_half:10.4f} pc, with loss = {result.fun:10.6f}")

    # Save the best fit r_{1/2}
    print("... saving best fit r_{1/2} " + f"to {file_path_r_half} (shape: {r_half.shape})")
    np.save(file_path_r_half, r_half)

    fr = r_half * np.arctan(r / r_half)  # shape (N,)
    dfr = r_half ** 2 / (r_half ** 2 + r ** 2)  # Derivative of f(r) with respect to r, shape (N,)
    del r, variances  # Free memory
    gc.collect()  # Force garbage collection

    # Save contracted distance
    print(f"... saving contracted distance to {file_path_fr} (shape: {fr.shape})")
    np.save(file_path_fr, fr)
    del fr  # Free memory
    gc.collect()  # Force garbage collection

    # Save contracted distance uncertainties
    print(f"... saving contracted distance uncertainties to {file_path_delta_fr} (shape: {dfr.shape}).\n")
    np.save(file_path_delta_fr, dfr)
    del dfr  # Free memory
    gc.collect()  # Force garbage collection

@njit()
def weighted_average_sym_kl_contracted(r_half, variances, r, weights):
    """
    Compute the weighted average symmetrized KL divergence between propagated 
    spherical coordinate uncertainties and an optimal isotropic Gaussian under a 
    contracted distance metric f(r) = r_half * arctan(r / r_half).

    Parameters:
        r_half : float
            Contraction scale parameter r_{1/2} (in pc).
        variances : np.ndarray of shape (N, 3)
            Each row contains uncertainties: (delta_r^2, delta_ra*^2, delta_dec^2)
            where delta_ra* = cos(dec) * delta_ra in radians.
        r : np.ndarray of shape (N,)
            Radial distances (in pc) for each source.
        weights : np.ndarray of shape (N,)
            Weights for each source in the average.

    Returns:
        alpha_opt : float
            Optimal scalar variance alpha.
        avg_kl_sym : float
            Average symmetrized KL divergence in contracted space.
    """
    # Unpack uncertainties of observables
    var_r, var_ra_star, var_dec = variances.T

    # Contracted distance and its derivative
    f_r = r_half * np.arctan(r / r_half)
    r_half_squared = r_half**2
    f_prime = r_half_squared / (r_half_squared + r**2)

    # First-order propagated variances (diagonal)
    var_los = f_prime**2 * var_r               # LOS direction
    var_tan1 = f_r**2 * var_ra_star            # horizontal tangential
    var_tan2 = f_r**2 * var_dec                # vertical tangential

    # Combine into diagonal covariance matrix for each source
    tr = var_los + var_tan1 + var_tan2
    tr_inv = 1.0 / var_los + 1.0 / var_tan1 + 1.0 / var_tan2

    # Weighted symmetrized KL divergence
    avg_kl_sym = np.average(np.sqrt(tr * tr_inv), weights=weights) - 3.0

    return avg_kl_sym

def calculate_contracted_data_and_errors_for_subsample(overwrite=False):
    """
    Calculate Cartesian positions and velocities, and their uncertainties, 
    under a contracted distance transform with zero radial velocity.
    """
    # Check if the contracted astrometric representation already exists
    file_path_position = os.path.join(SUBSAMPLE_PATH, "contracted_positions.npy")
    file_path_velocity = os.path.join(SUBSAMPLE_PATH, "contracted_velocities.npy")
    file_path_sigma_pos = os.path.join(SUBSAMPLE_PATH, "contracted_position_uncertainties.npy")
    file_path_sigma_vel = os.path.join(SUBSAMPLE_PATH, "contracted_velocity_uncertainties.npy")
    if os.path.exists(file_path_position) and os.path.exists(file_path_velocity) and os.path.exists(file_path_sigma_pos) and os.path.exists(file_path_sigma_vel) and not overwrite:
        print(f"Contracted astrometric representation and its uncertainties already exist at:")
        print(f"\t{file_path_position} ,")
        print(f"\t{file_path_velocity} ,")
        print(f"\t{file_path_sigma_pos} , and")
        print(f"\t{file_path_sigma_vel} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Computing contracted astrometric representation and uncertainties for subsample...")

    # Load required arrays
    print("... loading required arrays for positions and velocities")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    ra, dec = np.deg2rad(np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_equatorial_coordinates.npy"))[subsample_mask]).T  # shape (N, 2) in radians
    mu_ra, mu_dec = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_proper_motions.npy"))[subsample_mask].T  # shape (N, 2) in mas/yr
    
    # Unit vector in the direction of the star
    cos_ra, sin_ra = np.cos(ra), np.sin(ra)
    cos_dec, sin_dec = np.cos(dec), np.sin(dec)
    del ra, dec  # Free memory
    gc.collect()  # Force garbage collection

    # Load the contracted distance
    f_r = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_distance.npy"))  # (N,)

    # Positions
    print("... calculating transformed positions")
    positions = f_r[:, None] * np.column_stack([cos_ra * cos_dec, sin_ra * cos_dec, sin_dec])

    # Save the transformed positions
    print(f"... saving transformed positions to {file_path_position} (shape: {positions.shape})")
    np.save(file_path_position, positions)
    del positions  # Free memory
    gc.collect()  # Force garbage collection

    # Tangential velocity direction components
    print("... calculating transformed velocities")
    mu_ra_cos_dec = mu_ra * cos_dec  # shape (N,) in radians
    e_alpha = np.column_stack([-sin_ra, cos_ra, np.zeros_like(cos_ra)])  # Tangential basis vector in RA direction
    e_delta = np.column_stack([-cos_ra * sin_dec, -sin_ra * sin_dec, cos_dec])  # Tangential basis vector in Dec direction
    velocity = f_r[:, None] * (mu_ra_cos_dec[:, None] * e_alpha + mu_dec[:, None] * e_delta)

    # Save the transformed kinematics
    print(f"... saving transformed velocities to {file_path_velocity} (shape: {velocity.shape})")
    np.save(file_path_velocity, velocity)
    del e_alpha, e_delta, velocity  # Free memory
    gc.collect()  # Force garbage collection

    # Load uncertainties
    print("... loading observational uncertainties")
    f_r_prime = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_distance_error.npy"))  # (N,)
    astrometric_errors = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_astrometric_errors.npy"))[subsample_mask]  # (N, 5)
    sigma_ra, sigma_dec = np.deg2rad(astrometric_errors[:, :2]).T  # shape (N, 2) in radians
    sigma_mu_ra, sigma_mu_dec = astrometric_errors[:, 2:].T  # shape (N, 2) in radians
    lo, high = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_lo_high_photogeo.npy"))[subsample_mask].T  # each (N,) in pc
    sigma_r = (high - lo) / 2  # shape (N,) in pc
    del subsample_mask, astrometric_errors  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute some terms
    fr_sq = f_r**2  # shape (N,)
    mu_magnitude_sq = mu_ra_cos_dec**2 + mu_dec**2  # shape (N,) in radians
    frprime_sigmar_sq = (f_r_prime * sigma_r)**2  # shape (N,)

    # Position uncertainty
    print("... calculating position uncertainties")
    sigma_pos = np.sqrt(
        frprime_sigmar_sq +
        fr_sq * (
            (cos_dec * sigma_ra)**2 +
            sigma_dec**2
        )
    )

    # Save sigma_pos
    print(f"... saving position uncertainties to {file_path_sigma_pos} (shape: {sigma_pos.shape})")
    np.save(file_path_sigma_pos, sigma_pos)

    # Velocity uncertainty
    print("... calculating velocity uncertainties")
    sigma_vel = np.sqrt(
        frprime_sigmar_sq * mu_magnitude_sq +                           # Radial component
        fr_sq * (
            (mu_ra_cos_dec**2 + (mu_dec * sin_dec)**2) * sigma_ra**2 +  # Right ascension component
            ((mu_ra * sin_dec)**2 + mu_dec**2) * sigma_dec**2 +         # Declination component
            (cos_dec * sigma_mu_ra)**2 +                                # Proper motion in the right ascension component
            (sigma_mu_dec)**2                                           # Proper motion in the declination component
        )
    )
    del cos_dec, sin_dec, f_r, f_r_prime, sigma_r, sigma_ra, sigma_dec, sigma_mu_ra, sigma_mu_dec  # Free memory
    gc.collect()  # Force garbage collection

    # Save sigma_vel
    print(f"... saving velocity uncertainties to {file_path_sigma_vel} (shape: {sigma_vel.shape}).\n")
    np.save(file_path_sigma_vel, sigma_vel)

def construct_data_space_for_subsample(overwrite=False):
    """
    Calculate the data space for the subsample.
    """
    # Check if data space already exists
    file_data_space = os.path.join(SUBSAMPLE_PATH, "data_space.npy")
    if os.path.exists(file_data_space) and not overwrite:
        print(f"Data space already exists at:\n\t{file_data_space} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating data space for subsample...")

    # Load the contracted positions and velocities
    print("... loading required arrays")
    positions = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_positions.npy"))  # (N, 3)
    velocities = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_velocities.npy"))  # (N, 3)
    delta_pos = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_position_uncertainties.npy"))  # (N,)
    delta_vel = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_velocity_uncertainties.npy"))  # (N,)

    # Calculate scaling factor for positions
    norm_pos = np.median(delta_pos)  # Calculate scaling factor
    print(f"... scaling factor for positions: {norm_pos:.3f}")
    positions /= norm_pos  # Scale positions

    # Calculate scaling factor for velocities
    norm_vel = np.median(delta_vel)  # Calculate scaling factor
    print(f"... scaling factor for velocities: {norm_vel:.3f}")
    velocities /= norm_vel  # Scale velocities

    # Construct data space for clustering
    print("... constructing data space")
    data_space = np.concatenate([positions, velocities], axis=1)  # shape (N, 6)

    # Save data space
    print(f"... saving data space to {file_data_space} (shape: {data_space.shape}).\n")
    np.save(file_data_space, data_space)


# === Apply AstroLink to subsample and plot of cluster properties ===
def apply_astrolink_to_subsample(overwrite=False):
    """
    Run AstroLink clustering on the subsample.
    """
    # Check if AstroLink clustering output already exists
    file_astrolink_object = os.path.join(CLUSTERING_PATH, "astrolink_object.npz")
    if os.path.exists(file_astrolink_object) and not overwrite:
        print(f"AstroLink clustering output already exists at:\n\t{file_astrolink_object} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Running AstroLink clustering on the subsample...")

    # Load the required arrays
    print("... loading required arrays for AstroLink clustering")
    input_data = np.load(os.path.join(SUBSAMPLE_PATH, "data_space.npy"))  # (N, 6)
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    total_sf_mean = np.load(os.path.join(SUBSAMPLE_PATH, "total_selection_function_mean.npy"))[subsample_mask]  # (N,)
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Initialize AstroLink
    print("... initializing AstroLink object")
    clusterer = AstroLink(
        P=input_data,
        d_intrinsic=5,
        weights=1/total_sf_mean,
        k_den=KNN_FOR_ASTROLINK,
        adaptive=0,
        S=OPTIMAL_SIGMA_THRESHOLD,
        workers=MAX_PARALLEL_WORKERS,
        verbose=0
    )
    del input_data, total_sf_mean  # Free memory
    gc.collect()  # Force garbage collection

    # The following is a reworked version of the astrolink.run() method
    # It is more memory efficient and has print statements that better align with the rest of the script
    print(f"... [AstroLink] Started             | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    begin = time.perf_counter()

    # Transform the data (this doesn't do anything in this case, but is required to create the P_transform attribute)
    clusterer.transform_data()
    del clusterer.P  # Free memory
    gc.collect()  # Force garbage collection

    # Compute densities and nearest neighbours
    print(f"... [AstroLink] Computing densities and nearest neighbours\r")
    clusterer.estimate_density_and_kNN()
    del clusterer.weights  # Free memory
    gc.collect()  # Force garbage collection

    # Order points, find groups, compute prominences
    print(f"... [AstroLink] Aggregating points, finding groups, and computing prominences\r")
    clusterer.aggregate()

    # Fit model to subgroup prominences and find group significance
    print(f"... [AstroLink] Fitting model to subgroup prominences and finding group significances\r")
    clusterer.compute_significances()

    # Find clusters and hierarchy
    print(f"... [AstroLink] Finding clusters and their hierarchy                                  \r")
    clusterer.extract_clusters()

    clusterer._totalTime = time.perf_counter() - begin
    print(f"... [AstroLink] Completed           | {time.strftime('%Y-%m-%d %H:%M:%S')}       ")
    print(f"... [AstroLink] kNN query time      | {100*clusterer._logRhoTime/clusterer._totalTime:.2f}%       ")
    print(f"... [AstroLink] Aggregation time    | {100*clusterer._aggregateTime/clusterer._totalTime:.2f}%    ")
    print(f"... [AstroLink] Regression time     | {100*clusterer._regrTime/clusterer._totalTime:.2f}%         ")
    print(f"... [AstroLink] Rejection time      | {100*clusterer._rejTime/clusterer._totalTime:.2f}%          ")
    print(f"... [AstroLink] Total time          | {clusterer._totalTime:.2f} seconds!")
    print(f"... found {clusterer.clusters.shape[0] - 1} clusters at S={clusterer.S} in the clustering output")

    # Save the clustering output
    print(f"... saving AstroLink clustering output to {file_astrolink_object}.\n")
    saveAstroLinkObject(clusterer, file_astrolink_object)

def plot_prominence_model_fit(overwrite=False):
    """
    Plot the prominence model fit from AstroLink.
    """
    # Check if plots already exist
    file_prominence_model_fit_path = os.path.join(FIGURES_PATH, "prominence_model_fit.png")
    if os.path.exists(file_prominence_model_fit_path) and not overwrite:
        print(f"Prominence model fit plot already exists at:\n\t{file_prominence_model_fit_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting AstroLink prominence model fit...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    
    # Plot the prominence model fit
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot prominences histogram
    subgroup_proms = clusterer.prominences[:, 1]
    bw = 2*np.subtract(*np.percentile(subgroup_proms, [75, 25]))*subgroup_proms.size**(-1/3) # Freedman-Diaconis rule
    h, _, _ = ax.hist(
        subgroup_proms,
        bins=np.arange(np.ceil(subgroup_proms.max()/bw).astype(np.int64) + 1)*bw,
        density=True,
        histtype='stepfilled',
        facecolor=np.array([mcolors.to_rgba('k', alpha = 0.2)]),
        edgecolor='k',
        lw=1
    )

    # Plot fitted prominence model
    xs = np.linspace(0, subgroup_proms.max(), 10**4)
    ys = beta.pdf(xs, clusterer.pFit[0], clusterer.pFit[1])
    ax.plot(
        xs,
        ys,
        c='C0',
        lw=2,
        alpha=1.0,
        zorder=2,
        label='Noise model fit'
    )
    del subgroup_proms, xs, ys  # Free memory
    gc.collect()  # Force garbage collection

    # Add secondary x-axis showing significance levels
    def prom_to_sigma(prom):
        """Convert prominence -> significance."""
        prom = np.clip(prom, 1e-10, np.inf)  # avoid 0 or negative
        sf = beta.sf(prom, clusterer.pFit[0], clusterer.pFit[1])
        sf = np.clip(sf, 1e-300, 1 - 1e-16)  # avoid 0 or 1
        return norm.isf(sf)

    def sigma_to_prom(sigma):
        """Convert significance -> prominence."""
        sigma = np.clip(sigma, -10, 50)  # keep finite range
        sf = norm.sf(sigma)
        sf = np.clip(sf, 1e-300, 1 - 1e-16)
        return beta.isf(sf, clusterer.pFit[0], clusterer.pFit[1])

    ax_top = ax.secondary_xaxis('top', functions=(prom_to_sigma, sigma_to_prom))

    # Define tick positions
    sigma_ticks = np.arange(-4, 11)
    sigma_ticklabels = [f"{s:d}" for s in sigma_ticks]
    prom_zero = 0.0 # Add the special leftmost tick corresponding to prominence = 0.0 (use S = -np.inf for labeling)
    sigma_prom_zero = prom_to_sigma(1e-10)  # for position; ~very negative
    all_ticks = np.concatenate(([sigma_prom_zero], sigma_ticks))
    all_labels = [r"$-\infty$"] + ['']*4 + sigma_ticklabels[4:]

    # Apply ticks and labels
    ax_top.set_xticks(all_ticks)
    ax_top.set_xticklabels(all_labels)
    ax_top.set_xlim(ax.get_xlim())

    # Plot number of clusters as a function of significance
    ax_right = ax.twinx()  # create secondary y-axis on the right
    optimal_prominence = beta.isf(norm.sf(clusterer.S), clusterer.pFit[0], clusterer.pFit[1])
    optimal_num_clusters = clusterer.clusters.shape[0] - 1  # Exclude the background cluster
    prominences = np.empty_like(SIGMA_THRESHOLDS_FOR_COMPARISONS)
    num_clusters = np.empty_like(SIGMA_THRESHOLDS_FOR_COMPARISONS).astype(np.int64)
    for i, significance in enumerate(SIGMA_THRESHOLDS_FOR_COMPARISONS):
        # Extract clusters at this significance threshold
        clusterer.S = significance
        clusterer.extract_clusters()

        # Record prominence and number of clusters
        prominences[i] = beta.isf(norm.sf(significance), clusterer.pFit[0], clusterer.pFit[1])
        num_clusters[i] = clusterer.clusters.shape[0] - 1  # Exclude the background cluster

    # Plot curve (using the same x-scale as the main histogram)
    ax_right.plot(
        prominences,
        num_clusters,
        color='C1',
        lw=2,
        alpha=1.0,
        zorder=2,
        label="Number of clusters"
    )
    del prominences  # Free memory
    gc.collect()  # Force garbage collection

    # Adjust limits of axes
    ax.set_xlim(0, beta.isf(norm.sf(SIGMA_THRESHOLDS_FOR_COMPARISONS[-1]), clusterer.pFit[0], clusterer.pFit[1]))
    ax.set_ylim(h[h > 0].min() * 0.5, ax.get_ylim()[1])  # Set y-axis limits to avoid zero and very high values
    minN, maxN = num_clusters.min(), num_clusters.max()
    min_ylim, maxN_logunit = 10**np.floor(np.log10(minN)), 10**np.floor(np.log10(maxN))
    max_ylim = np.ceil(maxN / maxN_logunit) * maxN_logunit
    ax_right.set_ylim(min_ylim, max_ylim)

    # Add vertical and horizontal lines for optimal significance threshold and number of clusters at that threshold
    ax_right.plot(
        [optimal_prominence, optimal_prominence, ax_right.get_xlim()[1]],
        [ax_right.get_ylim()[1], optimal_num_clusters, optimal_num_clusters],
        color='C2',
        lw=1,
        ls='dashed',
        alpha=1.0,
        zorder=1,
    )
    ax_right.text(
        optimal_prominence,
        10**(0.98 * np.log10(ax_right.get_ylim()[1])),
        f"S = {OPTIMAL_SIGMA_THRESHOLD}",
        color="C2",
        fontsize=10,
        rotation=90,
        ha='right',
        va='top'
    )
    ax_right.text(
        0.98 * ax_right.get_xlim()[1],
        optimal_num_clusters,
        r"$N(S)$" + f" = {optimal_num_clusters}",
        color="C2",
        fontsize=10,
        ha='right',
        va='bottom'
    )
    del clusterer, num_clusters  # Free memory
    gc.collect()  # Force garbage collection
    
    # Convert vertical axes to logarithmic scale
    ax.set_yscale('log')
    ax_right.set_yscale('log')

    # Add labels to all axes
    ax.set_xlabel(r'Prominence, $p_{g_\leq}$')
    ax.set_ylabel('Probability Density')
    ax_top.set_xlabel(r"Significance, $S$")
    ax_right.set_ylabel(r"Number of clusters, $N(S)$")

    # Combine legends from both vertical axes into one
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_right.get_legend_handles_labels()
    all_lines = lines_1 + lines_2
    all_labels = labels_1 + labels_2
    ax.legend(all_lines, all_labels, loc='upper right', frameon=False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_prominence_model_fit_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved prominence model fit plot to {file_prominence_model_fit_path}.\n")

def plot_cluster_labels_on_sky(overwrite=False):
    """
    Plot the clustering output from AstroLink.
    """
    # Check if plots already exist
    file_clusters_on_sky_path = os.path.join(FIGURES_PATH, "clusters_on_sky.png")
    if os.path.exists(file_clusters_on_sky_path) and not overwrite:
        print(f"Clusters on sky plot already exists at:\n\t{file_clusters_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting AstroLink clusters on the sky...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians for Mollweide projection
    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)

     # Mollweide expects longitudes in the range [-pi, pi] and latitudes in the range [-pi/2, pi/2]
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1 # Invert x-axis for on-sky astro plot
    del longitude_wrap_bool  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot clusters on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    print("... plotting clusters on the sky")
    for i, clst in enumerate(clusterer.clusters[1:]):
        clusterMembers = clusterer.ordering[clst[0]:clst[1]]
        ax.scatter(
            *galactic_coordinates[clusterMembers].T,
            facecolor=f"C{i}", edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each cluster with a different color
    del clusterer, galactic_coordinates, clusterMembers  # Free memory
    gc.collect()  # Force garbage collection

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_clusters_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters on sky plot to {file_clusters_on_sky_path}.\n")

def plot_cluster_proper_motions_on_sky(overwrite=False):
    """
    Plot the proper motions of the clusters on the sky.
    """
    # Check if plots already exist
    file_proper_motions_on_sky_path = os.path.join(FIGURES_PATH, "proper_motions_on_sky.png")
    if os.path.exists(file_proper_motions_on_sky_path) and not overwrite:
        print(f"Proper motions on sky plot already exists at:\n\t{file_proper_motions_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting AstroLink clusters' proper motions on the sky...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    equatorial_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_equatorial_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    proper_motions = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_proper_motions.npy"))[subsample_mask]  # (N, 2) in mas/yr
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Reduce coordinates to subsample (and convert to angles to radians)
    l, b = np.deg2rad(galactic_coordinates).T
    l[l > np.pi] -= 2*np.pi
    ra, dec = np.deg2rad(equatorial_coordinates).T
    mu_ra, mu_dec = proper_motions.T
    del galactic_coordinates, equatorial_coordinates, proper_motions  # Free memory
    gc.collect()  # Force garbage collection

    # Define coordinate in ICRS (Equatorial J2000)
    print("... converting proper motions from equatorial to galactic coordinates")
    # Unit vectors in ICRS basis
    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    del ra, dec  # Free memory
    gc.collect()  # Force garbage collection

    # Unit vectors
    ra_hat = np.column_stack([
        -sin_ra,
         cos_ra,
         np.zeros_like(cos_ra)
    ])  # shape (N, 3)
    dec_hat = np.column_stack([
        -cos_ra * sin_dec,
        -sin_ra * sin_dec,
         cos_dec
    ])  # shape (N, 3)
    del sin_ra, cos_ra, sin_dec  # Free memory
    gc.collect()  # Force garbage collection

    # Proper motion Cartesian components in ICRS
    mu_ra_cosdec = mu_ra * cos_dec  # shape (N,)
    mu_icrs = ra_hat * mu_ra_cosdec[:, None] + dec_hat * mu_dec[:, None]  # shape (N, 3)
    del mu_ra, mu_dec, cos_dec, ra_hat, dec_hat, mu_ra_cosdec  # Free memory
    gc.collect()  # Force garbage collection

    # ICRS-to-Galactic rotation matrix (J2000)
    R = np.array([
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669]
    ])

    # Unit vectors in galactic basis
    sin_l, cos_l = np.sin(l), np.cos(l)
    sin_b, cos_b = np.sin(b), np.cos(b)

    # Unit vectors
    l_hat = np.column_stack([
        -sin_l,
        cos_l,
        np.zeros_like(cos_l)
    ])  # shape (N, 3) for l
    b_hat = np.column_stack([
        -cos_l * sin_b,
        -sin_l * sin_b,
         cos_b
    ])  # shape (N, 3) for b
    del sin_l, cos_l, sin_b, cos_b  # Free memory
    gc.collect()  # Force garbage collection

    # Proper motion Cartesian components in Galactic coordinates
    mu_gal = R.dot(mu_icrs.T).T  # shape (N, 3)

    # Extract proper motions in galactic system
    mu_l_cosb = np.sum(l_hat * mu_gal, axis=-1)  # shape (N,)
    mu_b = np.sum(b_hat * mu_gal, axis=-1)  # shape (N,)
    del mu_icrs, R, l_hat, b_hat, mu_gal  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate proper motion colours for plotting
    print("... calculating proper motion colours for plotting")
    mu_magnitude = np.sqrt(mu_l_cosb**2 + mu_b**2)  # Proper motion magnitude in mas/yr
    mu_magnitude = np.clip(mu_magnitude, 0, 20) / 20  # Clip to avoid extreme values
    mu_angle = np.arctan2(mu_b, mu_l_cosb)  # Proper motion angle in radians
    mu_angle = (mu_angle + np.pi) / (2 * np.pi)  # Shift to [0, 2*pi] range
    colours = mcolors.hsv_to_rgb(np.column_stack([mu_angle, mu_magnitude, np.ones_like(mu_angle)]))
    del mu_l_cosb, mu_b, mu_magnitude, mu_angle  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot clusters on the sky
    print("... creating Mollweide projection plot for clusters' proper motions on the sky")
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    for i, clst in enumerate(clusterer.clusters[1:]):
        clusterMembers = clusterer.ordering[clst[0]:clst[1]]
        ax.scatter(
            -l[clusterMembers], b[clusterMembers], # Invert x-axis for on-sky astro plot
            facecolor=colours[clusterMembers], edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each cluster with colours according to their proper motions
    del clusterer, l, b, colours  # Free memory
    gc.collect()  # Force garbage collection

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Tighten layout before adding inset axes
    plt.tight_layout()

    # Make the colour wheel for proper motions
    print("... creating colour wheel for proper motions")
    # Resolution of the colour wheel
    N = 256
    radius = 1
    y, x = np.ogrid[-radius:radius:N*1j, -radius:radius:N*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    del y, x  # Free memory
    gc.collect()  # Force garbage collection

    # Create HSV image
    hue = (theta + np.pi) / (2 * np.pi)        # [0, 1]
    saturation = np.clip(r, 0, 1)              # [0, 1]
    value = np.ones_like(hue)                  # fixed at 1
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    del hue, saturation, value, hsv  # Free memory
    gc.collect()  # Force garbage collection

    # Add alpha channel
    alpha = np.ones((N, N, 1))  # Shape (N, N, 1)
    rgba = np.concatenate([rgb, alpha], axis=-1)  # Shape (N, N, 4)
    del rgb, alpha  # Free memory
    gc.collect()  # Force garbage collection

    # Mask outside the circle
    mask = r > 1
    rgba[mask] = 0  # clear (alpha=0.0) outside the circle

    # Add inset axes
    size = 0.215  # Size of the inset axes as a fraction of the main axes
    fig_width, fig_height = fig.get_size_inches()
    base = min(fig_width, fig_height)
    width_abs = size * base  # inches
    height_abs = size * base
    loc = 4  # Location of the inset axes (4 = lower right corner)
    inset_ax = inset_axes(ax, width=width_abs, height=height_abs, loc=loc, borderpad=0)

    # Plot the colour wheel in the inset axes
    inset_ax.imshow(rgba[:, ::-1, :], extent=(-1, 1, -1, 1), origin='lower')
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_aspect('equal')
    del rgba  # Free memory
    gc.collect()  # Force garbage collection

    # Add text and lines to the inset axes
    inset_ax.text(-radius, 0, r"$+\mu_{l*}$", ha='right', va='center', fontsize=10, color='k')
    inset_ax.text(0, radius, r"$+\mu_{b}$", ha='center', va='bottom', fontsize=10, color='k')

    # Draw black border and proper motion circles
    circle = plt.Circle((0, 0), radius, color='k', fill=False, lw=1)
    inset_ax.add_patch(circle)
    inset_ax.text(-radius / np.sqrt(2), -radius / np.sqrt(2), r"$\geq20$ [mas/yr]", ha='right', va='top', fontsize=8, color='k')
    circle = plt.Circle((0, 0), 0.5 * radius, color='k', fill=False, lw=0.25)
    inset_ax.add_patch(circle)
    inset_ax.text(-0.5 * radius / np.sqrt(2), -0.5 * radius / np.sqrt(2), r"$10$", ha='right', va='top', fontsize=8, color='k')
    circle = plt.Circle((0, 0), 0.005, color='k', fill=False, lw=0.25)
    inset_ax.add_patch(circle)
    inset_ax.text(0, 0, r"$0$", ha='right', va='top', fontsize=8, color='k')

    # Hide inset axes elements
    inset_ax.set_frame_on(False)
    inset_ax.set_facecolor('none')
    inset_ax.patch.set_alpha(0.0)

    # Save the figure
    print(f"... saving proper motions on sky plot to {file_proper_motions_on_sky_path}.")
    plt.savefig(file_proper_motions_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved proper motions on sky plot to {file_proper_motions_on_sky_path}.\n")


# === Define reusable methods for comparing to existing catalogues ===
def load_cds_table(readme_path, data_path):
    """
    Load a CDS/VizieR fixed-width .dat.gz file into a pandas DataFrame
    using column specs parsed from the ReadMe file.

    Parameters
    ----------
    readme_path : str or Path
        Path to the CDS ReadMe file.
    data_path : str or Path
        Path to the .dat.gz file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with parsed columns.
    """
    readme_path = Path(readme_path)
    data_path = Path(data_path)
    table_name_nogz = data_path.name.replace(".gz", "")

    # Read ReadMe
    with open(readme_path, "r") as f:
        lines = f.readlines()

    # Find start of table section
    start_idx = None
    for i, line in enumerate(lines):
        if f"Byte-by-byte Description of file: {table_name_nogz}" in line:
            start_idx = i + 2  # Skip header line
            break
    if start_idx is None:
        raise ValueError(f"Table {table_name_nogz} not found in ReadMe.")

    colspecs = []
    names = []

    # Updated pattern: allow spaces around dash, dash optional
    pattern = re.compile(
        r"^\s*(\d+)(?:\s*-\s*(\d+))?\s+\S+\s+\S+\s+(\S+)"
    )

    for line in lines[start_idx:]:
        if not line.strip():
            break
        m = pattern.match(line)
        if m:
            start, end, name = m.groups()
            start = int(start)
            end = int(end) if end else start  # single column case
            colspecs.append((start - 1, end))
            names.append(name)

    # Read fixed-width file
    df = pd.read_fwf(data_path, compression="gzip", colspecs=colspecs, names=names)
    return df

def compare_to_catalogue_helper(members_cluster_ids_subsample,
                                members_cluster_probs_subsample,
                                cluster_probability_sums_total,
                                cluster_probability_sums_overlap):
    # Max cluster ID in the catalogue (this + 1 is the "no cluster" ID")
    max_catalogue_cluster_ID = members_cluster_ids_subsample.max() - 1

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    ordering = clusterer.ordering  # avoid sending the whole clusterer to workers

    # Put large arrays into shared memory
    print("... putting large arrays into shared memory")
    shm_ids, shape_ids, dtype_ids = arr_to_shared_memory(members_cluster_ids_subsample)
    shm_probs, shape_probs, dtype_probs = arr_to_shared_memory(members_cluster_probs_subsample)
    shm_ordering, shape_ordering, dtype_ordering = arr_to_shared_memory(ordering)

    # Calculate the RPJE values for each significance level
    whichClusters = -np.ones((SIGMA_THRESHOLDS_FOR_COMPARISONS.size, cluster_probability_sums_total.size, 2), dtype=np.int64)  # (N_clusters, 2) to store AstroLink clusters (start, end) pairs
    RPJE = np.zeros((SIGMA_THRESHOLDS_FOR_COMPARISONS.size, cluster_probability_sums_total.size, 2, 4), dtype=np.float32)  # (N_clusters, 4) to store RPJE values
    
    # Loop over significance values
    for k, significance in enumerate(SIGMA_THRESHOLDS_FOR_COMPARISONS):
        print(f"... calculating cluster-match statistics at significance level S={significance:.1f}   ", end = '\r')
        # Extract clusters at the current significance level
        clusterer.S = significance
        clusterer.extract_clusters()

        # Loop over AstroLink clusters in parallel
        max_workers = min(8, MAX_PARALLEL_WORKERS) # Use limited number of workers because this process is memory intensive
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, ((start, end), cluster_id) in enumerate(zip(clusterer.clusters, clusterer.ids)):
                if i == clusterer.ids.size - 1 or not clusterer.ids[i + 1].startswith(cluster_id + '-'): # Leaf clusters only
                    futures.append(
                        executor.submit(process_astrolink_cluster,
                                        start, end,
                                        cluster_probability_sums_total,
                                        cluster_probability_sums_overlap,
                                        max_catalogue_cluster_ID,
                                        shm_ids.name, shm_probs.name, shm_ordering.name,
                                        shape_ids, shape_probs, shape_ordering,
                                        dtype_ids, dtype_probs, dtype_ordering)
                    )

            for f in as_completed(futures):
                result = f.result()
                if result is None:
                    continue
                
                # Unpack result
                start, end, unique_ids, RPJE_cluster = result

                # Merge results back into global arrays
                better_matches = RPJE_cluster[:, 0, 2] > RPJE[k, unique_ids, 0, 2] # Jaccard index comparison under the full catalogue assumption
                which_better_matches = unique_ids[better_matches]
                whichClusters[k, which_better_matches] = start, end
                RPJE[k, which_better_matches] = RPJE_cluster[better_matches]
    
    # Clean up shared memory
    print("... cleaning up shared memory                                                                                                  ")
    shm_ids.close(); shm_ids.unlink()
    shm_probs.close(); shm_probs.unlink()
    shm_ordering.close(); shm_ordering.unlink()

    del clusterer, ordering  # Free memory
    gc.collect()  # Force garbage collection

    return whichClusters, RPJE

def arr_to_shared_memory(arr):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype.type, buffer=shm.buf)
    np.copyto(shm_arr, arr)
    return shm, arr.shape, arr.dtype.type

def process_astrolink_cluster(start, end,
                              cluster_probability_sums_total,
                              cluster_probability_sums_overlap,
                              max_cluster_ID,
                              shm_name_ids, shm_name_probs, shm_name_ordering,
                              shape_ids, shape_probs, shape_ordering,
                              dtype_ids, dtype_probs, dtype_ordering):
    """
    Worker function to process one AstroLink cluster.
    Reattaches shared-memory arrays, extracts cluster members, and computes RPJE stats.
    Returns (updates to whichClusters, RPJE).
    """
    # Reattach shared-memory arrays
    shm_ids = shared_memory.SharedMemory(name=shm_name_ids)
    shm_probs = shared_memory.SharedMemory(name=shm_name_probs)
    shm_ordering = shared_memory.SharedMemory(name=shm_name_ordering)

    members_cluster_ids_subsample = np.ndarray(shape_ids, dtype=dtype_ids, buffer=shm_ids.buf)
    members_probs_subsample = np.ndarray(shape_probs, dtype=dtype_probs, buffer=shm_probs.buf)
    ordering = np.ndarray(shape_ordering, dtype=dtype_ordering, buffer=shm_ordering.buf)

    # Get cluster members in the AstroLink output
    astrolink_cluster_members = ordering[start:end]

    # Get cluster IDs for those members
    xmatched_cluster_IDs = members_cluster_ids_subsample[astrolink_cluster_members]

    # Flatten IDs
    ids_flat = xmatched_cluster_IDs.ravel()

    # Handle "no cluster" ID = max_cluster_ID + 1
    mask_valid = ids_flat <= max_cluster_ID
    ids_valid = ids_flat[mask_valid]

    if ids_valid.size == 0:
        return None  # No valid clusters to compare

    # Get number of members in the AstroLink cluster when adjusted for the intersection with catalogue clusters
    N_i = end - start - np.sum(xmatched_cluster_IDs[:, 0] == max_cluster_ID + 1)

    # If there are clusters to compare to, get valid probabilities
    probs_in_astrolink_cluster = members_probs_subsample[astrolink_cluster_members]
    probs_flat = probs_in_astrolink_cluster.ravel()
    probs_valid = probs_flat[mask_valid]

    # Vectorized grouping: unique IDs and their summed probabilities
    unique_ids, inv = np.unique(ids_valid, return_inverse=True)
    M_sums = np.bincount(inv, weights=probs_valid)

    # Call numba-jitted function
    RPJE_cluster = calculate_rpje_for_astrolink_cluster_matches(
        cluster_probability_sums_total,
        cluster_probability_sums_overlap,
        unique_ids,
        M_sums,
        N_i,
        start,
        end
    )

    return (start, end, unique_ids, RPJE_cluster)

@njit()
def calculate_rpje_for_astrolink_cluster_matches(
        cluster_probability_sums_total,
        cluster_probability_sums_overlap,
        unique_ids,
        M_sums,
        N_i,
        start,
        end
    ):
    # Allocate small arrays to store the results
    RPJE = np.zeros((unique_ids.size, 2, 4), dtype=np.float32)

    # Probability mass of the clusters under the assumption that the catalogue clusters are found from a subsample of the union of the clusters themselves 
    Prob_sums_overlap = cluster_probability_sums_overlap[unique_ids]
    union_in_overlap = N_i + Prob_sums_overlap - M_sums

    # Probability mass of the clusters under the assumption that the catalogue clusters are found from the full catalogue
    Prob_sums_total = cluster_probability_sums_total[unique_ids]
    union_full = N_i + Prob_sums_total - M_sums

    # Calculate and store the recovery, purity, Jaccard index, and evidence values under full catalogue assumption (minimum values)
    RPJE[:, 0, 0] = M_sums / Prob_sums_total
    RPJE[:, 0, 1] = M_sums / N_i
    RPJE[:, 0, 2] = M_sums / union_full
    RPJE[:, 0, 3] = union_full / (end - start + Prob_sums_total - M_sums)

    # Calculate and store the recovery, purity, Jaccard index, and evidence values under union of clusters assumption (maximum values)
    RPJE[:, 1, 0] = M_sums / Prob_sums_overlap 
    RPJE[:, 1, 1] = M_sums / N_i    # Stays the same because the subsample used for AstroLink clustering is fixed
    RPJE[:, 1, 2] = M_sums / union_in_overlap
    RPJE[:, 1, 3] = union_in_overlap / (end - start + Prob_sums_total - M_sums)

    return RPJE

@contextlib.contextmanager
def printout_suppressor():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield


# === Compare clustering output to Hunt & Reffert (2024) ===
def prepare_Hunt2024_for_comparison(overwrite=False):
    """
    Prepare the data for comparison with Hunt & Reffert (2024).
    """
    # Check if files already exist
    file_path_clusters_names = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_names.npy")
    file_path_clusters_types = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_types.npy")
    file_path_clusters_snr = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_snr.npy")
    file_path_H24_members_cluster_ids_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_members_cluster_ids_subsample.npy")
    file_path_H24_members_cluster_probs_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_members_cluster_probs_subsample.npy")
    file_path_H24_clusters_probability_sums_total = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_probability_sums_total.npy")
    file_path_H24_clusters_probability_sums_overlap = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_probability_sums_overlap.npy")

    # Skip if all files exist and overwrite is False
    all_exist = (os.path.exists(file_path_clusters_names) and
                 os.path.exists(file_path_clusters_types) and
                 os.path.exists(file_path_clusters_snr) and
                 os.path.exists(file_path_H24_members_cluster_ids_subsample) and
                 os.path.exists(file_path_H24_members_cluster_probs_subsample) and
                 os.path.exists(file_path_H24_clusters_probability_sums_total) and
                 os.path.exists(file_path_H24_clusters_probability_sums_overlap))
    if all_exist and not overwrite:
        print("Hunt & Reffert (2024) reduced data already exists at:")
        print(f"\t{file_path_clusters_names} ,")
        print(f"\t{file_path_clusters_types} ,")
        print(f"\t{file_path_clusters_snr} ,")
        print(f"\t{file_path_H24_members_cluster_ids_subsample} ,")
        print(f"\t{file_path_H24_members_cluster_probs_subsample} ,")
        print(f"\t{file_path_H24_clusters_probability_sums_total} , and")
        print(f"\t{file_path_H24_clusters_probability_sums_overlap} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with Hunt & Reffert (2024)...")

    # Read the clusters.dat.gz file
    print("... loading clusters.dat.gz data")
    readme_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/ReadMe")
    clusters_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/clusters.dat.gz")
    df_clusters = load_cds_table(readme_path, clusters_path)

    # Save the names, types, and snr of the clusters
    print("... saving cluster information")
    H24_clusters_names = df_clusters["Name"].to_numpy()  # Cluster names
    H24_clusters_types = df_clusters["Type"].to_numpy()  # Cluster types
    H24_clusters_snr = df_clusters["CST"].to_numpy()  # Signal-to-noise ratio
    np.save(file_path_clusters_names, H24_clusters_names)
    np.save(file_path_clusters_types, H24_clusters_types)
    np.save(file_path_clusters_snr, H24_clusters_snr)
    del df_clusters, H24_clusters_names, H24_clusters_types, H24_clusters_snr  # Free memory
    gc.collect()  # Force garbage collection

    # Read the members.dat.gz file
    print("... loading members.dat.gz data")
    H24_members_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/members.dat.gz")
    df_members = load_cds_table(readme_path, H24_members_path)

    # Save the cluster IDs and probabilities
    print("... saving member information")
    H24_members_source_ids = df_members['GaiaDR3'].to_numpy()  # Source IDs of the members
    H24_members_cluster_ids = df_members["ID"].to_numpy()  # Cluster IDs
    H24_members_probs = df_members["Prob"].to_numpy()  # Membership probabilities
    del df_members  # Free memory
    gc.collect()  # Force garbage collection

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Load subsample mask
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)

    # Save the membership mask
    print("... making membership mask")
    indices = np.searchsorted(gdr3_source_ids, H24_members_source_ids) # Assumes gdr3_source_ids is sorted
    H24_members_mask = np.zeros_like(gdr3_source_ids, dtype=np.bool_)  # Create a mask of the same shape as gdr3_source_ids
    H24_members_mask[indices] = True  # Set the indices of the members to True
    del gdr3_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Get Hunt+2024 cluster IDs for this subsample (multiple columns since stars can be in multiple Hunt+2024 clusters)
    print("... getting Hunt+2024 cluster IDs and membership probabilities for the subsample in this work")
    max_appearances = np.unique(H24_members_source_ids, return_counts=True)[1].max()
    max_H24_cluster_ID = H24_members_cluster_ids.max()
    H24_members_cluster_ids_gdr3 = np.full((H24_members_mask.size, max_appearances), max_H24_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_H24_cluster_ID + 1, representing no cluster
    H24_members_cluster_probs_gdr3 = np.zeros((H24_members_mask.size, max_appearances), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
    H24_members_mask_where = np.where(H24_members_mask)[0]  # Use indices of members in the full catalogue from now on to do efficient slicing/indexing
    del H24_members_mask  # Free memory
    gc.collect()  # Force garbage collection
    
    # Assign cluster IDs / membership probabilities to each of the members
    unique_H24_source_ids, indices, counts = np.unique(H24_members_source_ids, return_index=True, return_counts=True)
    for num_count in range(1, max_appearances + 1):
        # Get the relative position of the members in the full catalogue for this count
        num_count_bool = counts == num_count

        # Get the cluster IDs and probabilities for the members with this count
        if num_count == 1:
            cluster_ids = H24_members_cluster_ids[indices[num_count_bool]][:, None]
            members_probs = H24_members_probs[indices[num_count_bool]][:, None]
        else: # There are not too many of these so what follows is efficient enough
            cluster_ids = np.zeros((num_count_bool.sum(), num_count), dtype=np.int64)
            members_probs = np.zeros((num_count_bool.sum(), num_count), dtype=np.float32)
            for i, sid in enumerate(unique_H24_source_ids[num_count_bool]):
                # Get the indices of the members with this source ID
                source_id_match = np.where(H24_members_source_ids == sid)[0]

                # Get the cluster IDs and probabilities for these members
                cluster_ids[i] = H24_members_cluster_ids[source_id_match]
                members_probs[i] = H24_members_probs[source_id_match]

        # Assign the cluster IDs and probabilities to the members
        H24_members_cluster_ids_gdr3[H24_members_mask_where[num_count_bool], :num_count] = cluster_ids
        H24_members_cluster_probs_gdr3[H24_members_mask_where[num_count_bool], :num_count] = members_probs

    H24_members_cluster_ids_subsample = H24_members_cluster_ids_gdr3[subsample_mask]
    H24_members_cluster_probs_subsample = H24_members_cluster_probs_gdr3[subsample_mask]
    del subsample_mask, H24_members_source_ids, H24_members_cluster_ids_gdr3, H24_members_cluster_probs_gdr3, H24_members_mask_where, unique_H24_source_ids, indices, counts  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total sum of probabilities for each Hunt+2024 cluster
    print("... pre-computing the total sum of probabilities for each Hunt+2024 cluster")
    H24_clusters_probability_sums_total = np.bincount(H24_members_cluster_ids.ravel(), 
                                        weights=H24_members_probs.ravel(),
                                        minlength=max_H24_cluster_ID + 1)  # (N_clusters,)
    del H24_members_cluster_ids, H24_members_probs  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the sum of probabilities for each Hunt+2024 cluster in the overlap with the subsample
    print("... pre-computing the total sum of probabilities for each Hunt+2024 cluster in the subsample of this work")
    H24_clusters_probability_sums_overlap = np.bincount(H24_members_cluster_ids_subsample.ravel(),
                                        weights=H24_members_cluster_probs_subsample.ravel(),
                                        minlength=max_H24_cluster_ID + 1)[:max_H24_cluster_ID + 1]  # (N_clusters,)

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_H24_members_cluster_ids_subsample, H24_members_cluster_ids_subsample)
    np.save(file_path_H24_members_cluster_probs_subsample, H24_members_cluster_probs_subsample)
    np.save(file_path_H24_clusters_probability_sums_total, H24_clusters_probability_sums_total)
    np.save(file_path_H24_clusters_probability_sums_overlap, H24_clusters_probability_sums_overlap)
    del H24_members_cluster_ids_subsample, H24_members_cluster_probs_subsample, H24_clusters_probability_sums_total, H24_clusters_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

def plot_Hunt2024_clusters_on_sky(overwrite=False):
    """
    Plot the Hunt & Reffert (2024) clusters on the sky.
    """
    # Check if plot already exists
    file_clusters_on_sky_path = os.path.join(FIGURES_PATH, "Hunt2024_clusters_on_sky.png")
    if os.path.exists(file_clusters_on_sky_path) and not overwrite:
        print(f"Hunt & Reffert (2024) clusters plot already exists at:\n\t{file_clusters_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Hunt & Reffert (2024) clusters...")

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians for Mollweide projection
    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)

     # Mollweide expects longitudes in the range [-pi, pi] and latitudes in the range [-pi/2, pi/2]
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1 # Invert x-axis for on-sky astro plot

    # Load the Hunt & Reffert (2024) clustering output
    print("... loading Hunt & Reffert (2024) catalogue")
    H24_members_cluster_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_members_cluster_ids_subsample.npy"))
    H24_members_cluster_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_members_cluster_probs_subsample.npy"))

    # Simplify the catalogue
    print("... simplifying Hunt & Reffert (2024) catalogue for plotting")
    plottable_bool = np.any(H24_members_cluster_probs_subsample > 0.5, axis=1)
    no_cluster_id = H24_members_cluster_ids_subsample.max()  # Define "no cluster" ID
    H24_members_cluster_ids_subsample = H24_members_cluster_ids_subsample[plottable_bool]
    galactic_coordinates = galactic_coordinates[plottable_bool]
    print(f"... {plottable_bool.sum()} plottable stars")
    del H24_members_cluster_probs_subsample, plottable_bool  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot clusters on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    print("... plotting clusters on the sky")
    for i in range(no_cluster_id):
        clusterMembers = np.any(H24_members_cluster_ids_subsample == i, axis=1)
        ax.scatter(
            *galactic_coordinates[clusterMembers].T,
            facecolor=f"C{i}", edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each cluster with a different color

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_clusters_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters on sky plot to {file_clusters_on_sky_path}.\n")

def compare_to_Hunt2024(overwrite=False):
    """
    Compare the clustering output to the Hunt & Reffert (2024).
    """
    # Check if comparison results already exist
    file_path_best_match_astrolink_clusters = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_best_match_astrolink_clusters.npy")
    file_path_cluster_rpje = os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_rpje.npy")
    
    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_cluster_rpje) and
                 os.path.exists(file_path_best_match_astrolink_clusters))
    if all_exist and not overwrite:
        print("Hunt & Reffert (2024) comparison results already exist at:")
        print(f"\t{file_path_best_match_astrolink_clusters} and")
        print(f"\t{file_path_cluster_rpje} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to Hunt & Reffert (2024)...")

    # Load required arrays
    print("... loading required arrays for comparison")
    H24_members_cluster_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_members_cluster_ids_subsample.npy"))  # (N,)
    H24_members_cluster_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_members_cluster_probs_subsample.npy"))  # (N,)
    H24_clusters_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_probability_sums_total.npy"))  # (N_clusters,)
    H24_clusters_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_probability_sums_overlap.npy"))  # (N_clusters,)

    # Make the comparison
    whichClusters, RPJE = compare_to_catalogue_helper(
        H24_members_cluster_ids_subsample,
        H24_members_cluster_probs_subsample,
        H24_clusters_probability_sums_total,
        H24_clusters_probability_sums_overlap
    )
    del H24_members_cluster_ids_subsample, H24_members_cluster_probs_subsample, H24_clusters_probability_sums_total, H24_clusters_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_best_match_astrolink_clusters, whichClusters)
    np.save(file_path_cluster_rpje, RPJE)
    del whichClusters, RPJE  # Free memory
    gc.collect()  # Force garbage collection

def plot_Hunt2024_comparison_results(overwrite=False):
    """
    Plot the results of the comparison between clustering output and Hunt & Reffert (2024),
    showing recovery, purity, and Jaccard index under both subsample assumptions ('full' 
    and 'union'), with hatched regions indicating bounds.
    """
    # Check if plot already exists
    file_path = os.path.join(FIGURES_PATH, "Hunt2024_comparison_results.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Hunt & Reffert (2024) comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Hunt & Reffert (2024) comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJE = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_rpje.npy"))  # (N_sigmas, N_clusters, 2, 4)
    H24_cluster_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_probability_sums_total.npy"))  # (N_clusters,)
    H24_cluster_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_probability_sums_overlap.npy"))  # (N_clusters,)
    coverage = H24_cluster_probability_sums_overlap / H24_cluster_probability_sums_total  # (N_clusters,)
    del H24_cluster_probability_sums_total, H24_cluster_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Load Hunt & Reffert (2024) cluster types
    print("... loading Hunt & Reffert (2024) cluster types")
    H24_cluster_types = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Hunt2024/hunt24_clusters_types.npy"), allow_pickle=True)  # (N_clusters,)

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # ========== COMBINED (o,m,g) CLUSTERS ==========
    print("... plotting (o,m,g) combined statistics vs significance level")

    mask = (H24_cluster_types != 'r') & (H24_cluster_types != 'd')

    # Extract per-statistic and per-assumption arrays
    R_full,  R_union  = RPJE[:, mask, 0, 0], RPJE[:, mask, 1, 0]
    P_full,  P_union  = RPJE[:, mask, 0, 1], RPJE[:, mask, 1, 1]
    J_full,  J_union  = RPJE[:, mask, 0, 2], RPJE[:, mask, 1, 2]
    #E_full,  E_union  = RPJE[:, mask, 0, 3], RPJE[:, mask, 1, 3]
    C = coverage[mask]  # (N_clusters,)

    # Evidence-weighted means
    Rbar_full  = np.sum(R_full * C, axis=1) / np.sum(C)
    Rbar_union = np.sum(R_union * C, axis=1) / np.sum(C)
    Pbar_full  = np.sum(P_full * C, axis=1) / np.sum(C)
    Pbar_union = np.sum(P_union * C, axis=1) / np.sum(C)
    Jbar_full  = np.sum(J_full * C, axis=1) / np.sum(C)
    Jbar_union = np.sum(J_union * C, axis=1) / np.sum(C)

    """
    # Recovery
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_union,
            color='k', linestyle='dashed', linewidth=1.5, label='R (o,m,g)', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full,
            color='k', linestyle='dashed', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full, Rbar_union,
                    facecolor='none', hatch='//', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Purity
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_union,
            color='k', linestyle='dotted', linewidth=1.5, label='P (o,m,g)', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full,
            color='k', linestyle='dotted', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full, Pbar_union,
                    facecolor='none', hatch='\\', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
            color='k', linestyle='solid', linewidth=1.5, label='J (o,m,g)', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
            color='k', linestyle='solid', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                    color='k', alpha=0.3, zorder=3)
    """

    # ========== INDIVIDUAL CLUSTER TYPES ==========
    print("... plotting per-cluster-type statistics")
    cluster_type_and_colour = dict(zip(['o', 'm', 'g', 'd', 'r'], ['C0', 'C2', 'C1', 'C4', 'C3']))

    for cluster_type, type_colour in cluster_type_and_colour.items():
        mask = H24_cluster_types == cluster_type
        if not np.any(mask):
            continue

        #E_full,  E_union  = RPJE[:, mask, 0, 3], RPJE[:, mask, 1, 3]
        J_full,  J_union  = RPJE[:, mask, 0, 2], RPJE[:, mask, 1, 2]
        C = coverage[mask]  # (N_clusters,)

        # Weighted means
        Jbar_full  = np.sum(J_full * C, axis=1) / np.sum(C)
        Jbar_union = np.sum(J_union * C, axis=1) / np.sum(C)

        # Plot both bounds + hatched region
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
                color=type_colour, linestyle='solid', linewidth=0.75, alpha=0.8,
                label=f"J ({cluster_type})", zorder=2)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
                color=type_colour, linestyle='solid', linewidth=0.75, alpha=0.8, zorder=2)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                        color=type_colour, alpha=0.3, zorder=2)

        # Plot fraction of clusters of this type matched to above J=0.5
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, np.mean(J_union > 0.5, axis=1),
                color=type_colour, linestyle='dashed', linewidth=0.75, alpha=0.8,
                label=f"N({cluster_type} | J > 0.5)/N({cluster_type})", zorder=2)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, np.mean(J_full > 0.5, axis=1),
                color=type_colour, linestyle='dashed', linewidth=0.75, alpha=0.8, zorder=2)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, np.mean(J_full > 0.5, axis=1), np.mean(J_union > 0.5, axis=1),
                        facecolor='none', hatch='//', edgecolor=type_colour, linewidth=0.0, alpha=0.3, zorder=2)

    # ========== Final formatting ==========
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Significance Level")
    ax.set_ylabel("Comparison Statistic")
    ax.legend()

    print("... saving figure.\n")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Compare clustering output to the Unified Cluster Catalogue ===
def prepare_UCC_for_comparison(overwrite=False):
    """
    Prepare the data for comparison with the Unified Cluster Catalogue.
    """
    # Check if files already exist
    file_path_clusters_names = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_names.npy")
    file_path_clusters_quality_class = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_quality_class.npy")
    file_path_UCC_members_cluster_ids_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_members_cluster_ids_subsample.npy")
    file_path_UCC_members_cluster_probs_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_members_cluster_probs_subsample.npy")
    file_path_UCC_clusters_probability_sums_total = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_probability_sums_total.npy")
    file_path_UCC_clusters_probability_sums_overlap = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_probability_sums_overlap.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_clusters_names) and
                 os.path.exists(file_path_clusters_quality_class) and
                 os.path.exists(file_path_UCC_members_cluster_ids_subsample) and
                 os.path.exists(file_path_UCC_members_cluster_probs_subsample) and
                 os.path.exists(file_path_UCC_clusters_probability_sums_total) and
                 os.path.exists(file_path_UCC_clusters_probability_sums_overlap))
    if all_exist and not overwrite:
        print("Unified Cluster Catalogue reduced data already exists at:")
        print(f"\t{file_path_clusters_names} ,")
        print(f"\t{file_path_clusters_quality_class} ,")
        print(f"\t{file_path_UCC_members_cluster_ids_subsample} ,")
        print(f"\t{file_path_UCC_members_cluster_probs_subsample} ,")
        print(f"\t{file_path_UCC_clusters_probability_sums_total} , and")
        print(f"\t{file_path_UCC_clusters_probability_sums_overlap} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with the Unified Cluster Catalogue...")

    # Read the UCC_cat.csv file
    print("... loading UCC_cat.csv data")
    cat_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/UCC_cat.csv")
    df_cat = pd.read_csv(cat_path)

    # Save the cluster names in the same format as it appears in the UCC_members.parquet file
    print("... saving cluster names")
    UCC_clusters_names = df_cat["ID"].to_numpy().astype(np.str_)  # ';'-separated cluster names
    UCC_clusters_names = np.array([x[0] for x in np.char.split(UCC_clusters_names, ';')])  # Remove everything from the first ';' onwards
    UCC_clusters_names = np.char.replace(UCC_clusters_names, '+', 'p') # Replace '+' with 'p'
    UCC_clusters_names = np.array([re.sub(r'[^A-Za-z0-9]', '', x) for x in UCC_clusters_names]) # Remove non-alphanumeric characters
    UCC_clusters_names = np.char.lower(UCC_clusters_names) # Make lowercase
    np.save(file_path_clusters_names, UCC_clusters_names)
    del UCC_clusters_names  # Free memory
    gc.collect()  # Force garbage collection

    # Save the combined quality class of the clusters
    print("... saving cluster quality classes")
    UCC_clusters_quality_class = df_cat['C3'].to_numpy().astype(np.str_)  # Signal-to-noise ratio
    np.save(file_path_clusters_quality_class, UCC_clusters_quality_class)
    del df_cat, UCC_clusters_quality_class  # Free memory
    gc.collect()  # Force garbage collection

    # Read the UCC_members.parquet file
    print("... loading UCC_members.parquet data")
    UCC_members_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/UCC_members.parquet")
    df_members = pd.read_parquet(UCC_members_path)
    UCC_members_cluster_names = df_members["name"].to_numpy().astype(np.str_)  # Cluster names
    UCC_members_probs = df_members["probs"].to_numpy()  # Membership probabilities
    UCC_members_source_ids = df_members['Source'].to_numpy()  # Source IDs of the members
    del df_members  # Free memory
    gc.collect()  # Force garbage collection

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Save the membership mask
    print("... making membership mask")
    indices = np.searchsorted(gdr3_source_ids, UCC_members_source_ids) # Assumes gdr3_source_ids is sorted
    UCC_members_mask = np.zeros_like(gdr3_source_ids, dtype=np.bool_)  # Create a mask of the same shape as gdr3_source_ids
    UCC_members_mask[indices] = True  # Set the indices of the members to True
    del gdr3_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Load required arrays
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N_gdr3,)

    # Get UCC cluster IDs for this subsample (multiple columns since stars can be in multiple UCC clusters)
    print("... getting UCC cluster IDs and membership probabilities for the subsample in this work")
    max_appearances = np.unique(UCC_members_source_ids, return_counts=True)[1].max()
    unique_UCC_members_cluster_names, UCC_members_cluster_ids = np.unique(UCC_members_cluster_names, return_inverse=True)
    max_UCC_cluster_ID = UCC_members_cluster_ids.max()
    UCC_members_cluster_ids_gdr3 = np.full((UCC_members_mask.size, max_appearances), max_UCC_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_UCC_cluster_ID + 1, representing no cluster
    UCC_members_cluster_probs_gdr3 = np.zeros((UCC_members_mask.size, max_appearances), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
    UCC_members_mask_where = np.where(UCC_members_mask)[0]  # Use indices of members in the full catalogue from now on to do efficient slicing/indexing
    del UCC_members_mask, UCC_members_cluster_names, unique_UCC_members_cluster_names  # Free memory
    gc.collect()  # Force garbage collection

    # Assign the cluster IDs / membership probabilities to each of the members
    unique_UCC_source_ids, indices, counts = np.unique(UCC_members_source_ids, return_index=True, return_counts=True)
    for num_count in range(1, max_appearances + 1):
        # Get the relative position of the members in the full catalogue for this count
        num_count_bool = counts == num_count

        # Get the cluster IDs and probabilities for the members with this count
        if num_count == 1:
            cluster_ids = UCC_members_cluster_ids[indices[num_count_bool]][:, None]
            members_probs = UCC_members_probs[indices[num_count_bool]][:, None]
        else: # There are not too many of these so what follows is efficient enough
            cluster_ids = np.zeros((num_count_bool.sum(), num_count), dtype=np.int64)
            members_probs = np.zeros((num_count_bool.sum(), num_count), dtype=np.float32)
            for i, sid in enumerate(unique_UCC_source_ids[num_count_bool]):
                # Get the indices of the members with this source ID
                source_id_match = np.where(UCC_members_source_ids == sid)[0]

                # Get the cluster IDs and probabilities for these members
                cluster_ids[i] = UCC_members_cluster_ids[source_id_match]
                members_probs[i] = UCC_members_probs[source_id_match]

        # Assign the cluster IDs and probabilities to the members
        UCC_members_cluster_ids_gdr3[UCC_members_mask_where[num_count_bool], :num_count] = cluster_ids
        UCC_members_cluster_probs_gdr3[UCC_members_mask_where[num_count_bool], :num_count] = members_probs

    UCC_members_cluster_ids_subsample = UCC_members_cluster_ids_gdr3[subsample_mask]
    UCC_members_cluster_probs_subsample = UCC_members_cluster_probs_gdr3[subsample_mask]
    del subsample_mask, UCC_members_source_ids, UCC_members_cluster_ids_gdr3, UCC_members_cluster_probs_gdr3, UCC_members_mask_where, unique_UCC_source_ids, indices, counts  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total sum of probabilities for each UCC cluster
    print("... pre-computing the total sum of probabilities for each UCC cluster")
    UCC_cluster_probability_sums_total = np.bincount(UCC_members_cluster_ids, 
                                        weights=UCC_members_probs,
                                        minlength=max_UCC_cluster_ID + 1)  # (N_clusters,)
    del UCC_members_cluster_ids, UCC_members_probs  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the sum of probabilities for each UCC cluster in the overlap with the subsample
    print("... pre-computing the total sum of probabilities for each UCC cluster in the subsample of this work")
    UCC_cluster_probability_sums_overlap = np.bincount(UCC_members_cluster_ids_subsample.ravel(),
                                        weights=UCC_members_cluster_probs_subsample.ravel(),
                                        minlength=max_UCC_cluster_ID + 1)[:max_UCC_cluster_ID + 1]  # (N_clusters,)
    
    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_UCC_members_cluster_ids_subsample, UCC_members_cluster_ids_subsample)
    np.save(file_path_UCC_members_cluster_probs_subsample, UCC_members_cluster_probs_subsample)
    np.save(file_path_UCC_clusters_probability_sums_total, UCC_cluster_probability_sums_total)
    np.save(file_path_UCC_clusters_probability_sums_overlap, UCC_cluster_probability_sums_overlap)
    del UCC_members_cluster_ids_subsample, UCC_members_cluster_probs_subsample, UCC_cluster_probability_sums_total, UCC_cluster_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

def plot_UCC_clusters_on_sky(overwrite=False):
    """
    Plot the Unified Cluster Catalogue clusters on the sky.
    """
    # Check if plot already exists
    file_clusters_on_sky_path = os.path.join(FIGURES_PATH, "UCC_clusters_on_sky.png")
    if os.path.exists(file_clusters_on_sky_path) and not overwrite:
        print(f"Unified Cluster Catalogue clusters plot already exists at:\n\t{file_clusters_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Unified Cluster Catalogue clusters...")

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians for Mollweide projection
    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)

     # Mollweide expects longitudes in the range [-pi, pi] and latitudes in the range [-pi/2, pi/2]
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1 # Invert x-axis for on-sky astro plot

    # Load the UCC clustering output
    print("... loading Unified Cluster Catalogue catalogue")
    UCC_members_cluster_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_members_cluster_ids_subsample.npy"))
    UCC_members_cluster_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_members_cluster_probs_subsample.npy"))

    # Simplify the catalogue
    print("... simplifying Unified Cluster Catalogue catalogue for plotting")
    plottable_bool = np.any(UCC_members_cluster_probs_subsample > 0.5, axis=1)
    no_cluster_id = UCC_members_cluster_ids_subsample.max()  # Define "no cluster" ID
    UCC_members_cluster_ids_subsample = UCC_members_cluster_ids_subsample[plottable_bool]
    galactic_coordinates = galactic_coordinates[plottable_bool]
    print(f"... {plottable_bool.sum()} plottable stars")
    del UCC_members_cluster_probs_subsample, plottable_bool  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot clusters on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    print("... plotting clusters on the sky")
    for i in range(no_cluster_id):
        clusterMembers = np.any(UCC_members_cluster_ids_subsample == i, axis=1)
        ax.scatter(
            *galactic_coordinates[clusterMembers].T,
            facecolor=f"C{i}", edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each cluster with a different color

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_clusters_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters on sky plot to {file_clusters_on_sky_path}.\n")

def compare_to_UCC(overwrite=False):
    """
    Compare the clustering output to the Unified Cluster Catalogue.
    """
    # Check if comparison results already exist
    file_path_best_match_astrolink_clusters = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_best_match_astrolink_clusters.npy")
    file_path_cluster_rpje = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_rpje.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_best_match_astrolink_clusters) and
                 os.path.exists(file_path_cluster_rpje))
    if all_exist and not overwrite:
        print("Unified Cluster Catalogue comparison results already exist at:")
        print(f"\t{file_path_best_match_astrolink_clusters} and")
        print(f"\t{file_path_cluster_rpje} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to the Unified Cluster Catalogue...")

    # Load the reduced UCC data
    print("... loading reduced UCC data")
    UCC_members_cluster_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_members_cluster_ids_subsample.npy"))  # (N,)
    UCC_members_cluster_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_members_cluster_probs_subsample.npy"))  # (N,)
    UCC_clusters_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_probability_sums_total.npy"))  # (N_clusters,)
    UCC_clusters_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_probability_sums_overlap.npy"))  # (N_clusters,)

    # Make the comparison
    whichClusters, RPJE = compare_to_catalogue_helper(
        UCC_members_cluster_ids_subsample,
        UCC_members_cluster_probs_subsample,
        UCC_clusters_probability_sums_total,
        UCC_clusters_probability_sums_overlap
    )
    del UCC_members_cluster_ids_subsample, UCC_members_cluster_probs_subsample, UCC_clusters_probability_sums_total, UCC_clusters_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_best_match_astrolink_clusters, whichClusters)
    np.save(file_path_cluster_rpje, RPJE)
    del whichClusters, RPJE  # Free memory
    gc.collect()  # Force garbage collection

def plot_UCC_comparison_results(overwrite=False):
    """
    Plot the results of the comparison between clustering output and the Unified Cluster Catalogue (UCC),
    showing recovery, purity, and Jaccard index under both subsample assumptions
    ('full' and 'union'), with hatched regions indicating the bounds between them.
    """
    # Check if plot already exists
    file_path = os.path.join(FIGURES_PATH, "UCC_comparison_results.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Unified Cluster Catalogue comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Unified Cluster Catalogue comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJE = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_rpje.npy"))  # (N_sigmas, N_clusters, 2, 4)
    UCC_cluster_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_probability_sums_total.npy"))  # (N_clusters,)
    UCC_cluster_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_probability_sums_overlap.npy"))  # (N_clusters,)
    coverage = UCC_cluster_probability_sums_overlap / UCC_cluster_probability_sums_total  # (N_clusters,)
    del UCC_cluster_probability_sums_total, UCC_cluster_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Load UCC cluster metadata
    print("... loading Unified Cluster Catalogue cluster names and quality classes")
    UCC_clusters_names = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_names.npy"))
    UCC_clusters_quality_class = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC/ucc_clusters_quality_class.npy"))

    # Reorder the quality classes by the sorted names to match RPJE
    reorder = np.argsort(UCC_clusters_names)
    UCC_clusters_names = UCC_clusters_names[reorder]
    UCC_clusters_quality_class = UCC_clusters_quality_class[reorder]

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # ========== ALL CLUSTERS ==========
    print("... plotting overall cluster-match statistics vs significance level")

    # Extract per-assumption and per-statistic arrays
    R_full,  R_union  = RPJE[:, :, 0, 0], RPJE[:, :, 1, 0]
    P_full,  P_union  = RPJE[:, :, 0, 1], RPJE[:, :, 1, 1]
    J_full,  J_union  = RPJE[:, :, 0, 2], RPJE[:, :, 1, 2]
    #E_full,  E_union  = RPJE[:, :, 0, 3], RPJE[:, :, 1, 3]

    # Evidence-weighted means
    Rbar_full  = np.sum(R_full * coverage, axis=1) / np.sum(coverage)
    Rbar_union = np.sum(R_union * coverage, axis=1) / np.sum(coverage)
    Pbar_full  = np.sum(P_full * coverage, axis=1) / np.sum(coverage)
    Pbar_union = np.sum(P_union * coverage, axis=1) / np.sum(coverage)
    Jbar_full  = np.sum(J_full * coverage, axis=1) / np.sum(coverage)
    Jbar_union = np.sum(J_union * coverage, axis=1) / np.sum(coverage)

    # Recovery
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_union,
            color='k', linestyle='dashed', linewidth=1.5, label='R (all)', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full,
            color='k', linestyle='dashed', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full, Rbar_union,
                    facecolor='none', hatch='//', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Purity
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_union,
            color='k', linestyle='dotted', linewidth=1.5, label='P (all)', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full,
            color='k', linestyle='dotted', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full, Pbar_union,
                    facecolor='none', hatch='\\', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
            color='k', linestyle='solid', linewidth=1.5, label='J (all)', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
            color='k', linestyle='solid', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                    color='k', alpha=0.3, zorder=3)

    # ========== QUALITY CLASS GROUPS ==========
    print("... plotting statistics vs significance level for UCC quality ranges")

    cluster_class_lists = [
        ['AA', 'AB', 'BA'],
        ['AC', 'BB', 'CA'],
        ['AD', 'BC', 'CB', 'DA'],
        ['BD', 'CC', 'DB'],
        ['CD', 'DC', 'DD']
    ]
    cmap = plt.get_cmap('coolwarm')
    cluster_class_colours = [cmap(i / (len(cluster_class_lists) - 1)) for i in range(len(cluster_class_lists))]

    for class_list, colour in zip(cluster_class_lists, cluster_class_colours):
        mask = np.zeros(RPJE.shape[1], dtype=bool)
        for qclass in class_list:
            mask |= (UCC_clusters_quality_class == qclass)
        if not np.any(mask):
            continue

        # Extract Jaccard + evidence per assumption
        #E_full,  E_union  = RPJE[:, mask, 0, 3], RPJE[:, mask, 1, 3]
        J_full,  J_union  = RPJE[:, mask, 0, 2], RPJE[:, mask, 1, 2]
        C = coverage[mask]  # (N_clusters,)

        # Weighted means
        Jbar_full  = np.sum(J_full * C, axis=1) / np.sum(C)
        Jbar_union = np.sum(J_union * C, axis=1) / np.sum(C)

        # Plot both bounds + hatched region
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
                color=colour, linestyle='solid', linewidth=0.75, alpha=0.8,
                label='J (' + ','.join(class_list) + ')', zorder=2)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
                color=colour, linestyle='solid', linewidth=0.75, alpha=0.8, zorder=2)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                        color=colour, alpha=0.3, zorder=2)

    # ========== Final formatting ==========
    print('... saving figure.\n')
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Significance Level")
    ax.set_ylabel("Comparison Statistic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Compare clustering output to galstreams ===
def prepare_galstreams_for_comparison(overwrite=False):
    """
    Prepare the data for comparison with the galstreams catalogue.
    """
    # Check if galstreams folder exists in AUXILLARY_CATALOGUES_PATH
    galstreams_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams")
    os.makedirs(galstreams_path, exist_ok=True)
    
    # Check if files already exist
    file_path_galstreams_members_stream_ids_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_members_stream_ids_subsample.npy")
    file_path_galstreams_members_stream_probs_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_members_stream_probs_subsample.npy")
    file_path_galstreams_streams_probability_sums_total = os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_streams_probability_sums_total.npy")
    file_path_galstreams_streams_probability_sums_overlap = os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_streams_probability_sums_overlap.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_galstreams_members_stream_ids_subsample) and
                 os.path.exists(file_path_galstreams_members_stream_probs_subsample) and
                 os.path.exists(file_path_galstreams_streams_probability_sums_total) and
                 os.path.exists(file_path_galstreams_streams_probability_sums_overlap))
    if all_exist and not overwrite:
        print("Galstreams reduced data already exists at:")
        print(f"\t{file_path_galstreams_members_stream_ids_subsample} ,")
        print(f"\t{file_path_galstreams_members_stream_probs_subsample} ,")
        print(f"\t{file_path_galstreams_streams_probability_sums_total} , and")
        print(f"\t{file_path_galstreams_streams_probability_sums_overlap} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with the galstreams catalogue...")

    # Load required arrays
    print('... loading required arrays')
    source_ids = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_source_ids.npy"))  # (N_gdr3,)
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, 'subsample_mask.npy'))  # (N_gdr3,)
    ra, dec = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_equatorial_coordinates.npy")).T  # Each (N_gdr3,) in degrees
    mu_ra, mu_dec = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_proper_motions.npy")).T  # Each (N_gdr3,) in mas/yr
    #r_med_geo = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_med_geo.npy"))  # (N_gdr3,) in pc
    r_med_photogeo = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_med_photogeo.npy"))  # (N_gdr3,) in pc

    # Get MWStreams object from galstreams
    print('... creating MWStreams object')
    with printout_suppressor():  # Suppress the printout from galstreams
        mws = galstreams.MWStreams(print_topcat_friendly_files=False)

    # Manual fixes for missing width_phi2 values from galstreams
    mws.summary.loc['M30-S20', 'width_phi2'] = 0.10992290189439437 # From galstreams/tracks/track.st.M30.sollima2020.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['M5-G19', 'width_phi2'] = 0.00059498941000224  # From galstreams/tracks/track.st.M5.grillmair2019.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['NGC5053-L06', 'width_phi2'] = 0.5  # Based on Lauchner et al. (2006) maps and the cluster’s distance (~17.4 kpc), an angular width of ~0.5 deg is used a reasonable proxy.
    mws.summary.loc['NGC6362-S20', 'width_phi2'] = 0.03279231365173159 # From galstreams/tracks/track.st.NGC6362.sollima2020.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['Orphan-K23', 'width_phi2'] = 0.5610732369318953 # From galstreams/tracks/track.st.Orphan-Chenab.ibata2021.summary.ecsv (default track summary file has ~0 deg width)
    mws.summary.loc['Pal5-PW19', 'width_phi2'] = 0.30188212290570887 # From galstreams/tracks/track.st.Pal5.ibata2024.summary.ecsv (default track summary file has ~0 deg width)
    mws.summary.loc['Parallel-W18', 'width_phi2'] = 25  # Reasonable value considering Weiss et al. (2018) reported physical dispersions sigma (kpc) from SDSS fits
    mws.summary.loc['Perpendicular-W18', 'width_phi2'] = 25  # Reasonable value considering Weiss et al. (2018) reported physical dispersions sigma (kpc) from SDSS fits

    # Manual fixes for missing width_pm_phi1_cosphi2 and width_pm_phi2 values from galstreams
    mws.summary.loc['ACS-R21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.051429218098668614,0.060361599720383706]  # From galstreams/tracks/track.st.ACS.ramos2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['Gaia-10-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.25549095266698924,0.21102160186354021]  # From galstreams/tracks/track.st.Gaia-10.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Gaia-12-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.31870038997481237,0.39533356136581626]  # From galstreams/tracks/track.st.Gaia-12.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Gaia-6-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.4852100507178606,0.3356251329783215]  # From galstreams/tracks/track.st.Gaia-6.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Gaia-7-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.3693734621270431,0.30817705138525747]  # From galstreams/tracks/track.st.Gaia-7.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Hrid-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.8130582153516146,0.5597190294200691]  # From galstreams/tracks/track.st.Hrid.ibata2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['Jhelum-a-B19', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.08724815957044693,0.17890077278048597]  # From galstreams/tracks/track.st.Jhelum-a.shipp2019.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Jhelum-b-B19', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.12468187056175481,0.12024080502075932]  # From galstreams/tracks/track.st.Jhelum-b.shipp2019.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['M2-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.11258154325759534,0.09524201623728742]  # From galstreams/tracks/track.st.M2.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Monoceros-R21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.024488308491996437,0.025432522875930966]  # From galstreams/tracks/track.st.Monoceros.ramos2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['NGC1261-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.07311649510540366,0.009119289396829662] # From galstreams/tracks/track.st.NGC1261.ibata2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['NGC1851-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.15733960159151045,0.26652779597474907]  # From galstreams/tracks/track.st.NGC1851.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['NGC2298-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.3019327576574179,0.28430356991838157]  # From galstreams/tracks/track.st.NGC2298.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['NGC288-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.37797158611441367,0.2383575166183864]  # From galstreams/tracks/track.st.NGC288.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['OmegaCen-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [1.3122549027325625,1.7837014357475944]  # From galstreams/tracks/track.st.OmegaCen-Fimbulthul.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Orphan-K23', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.26468901721952365,0.3147176316063631]  # From galstreams/tracks/track.st.Orphan-Chenab.ibata2021.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Pal5-PW19', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.15042483606050822,0.16169990020590352]  # From galstreams/tracks/track.st.Pal5.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Ravi-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['SGP-S-Y22', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.24098786032510924,0.42219519260381305]  # From galstreams/tracks/track.st.SGP-S.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Turbio-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['Wambelong-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['Willka_Yaku-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['Yangtze-Y23', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors

    # Cycle through each stream, find which stars are in its footprint, and then compute stream membership probability with the information available
    max_galstream_cluster_ID = len(mws) - 1
    galstreams_members_stream_ids_gdr3 = np.full((subsample_mask.size, 1), max_galstream_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_galstreams_stream_ID + 1, representing no stream
    galstreams_members_stream_probs_gdr3 = np.zeros((subsample_mask.size, 1), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
    for i, (stream_track_name, stream) in enumerate(mws.items()):
        print(f'... calculating membership probabilities for stream {i + 1}/{len(mws)}: {stream_track_name}                   ', end='\r')
        # Width and sigma in phi2
        width_phi2 = mws.summary.loc[stream_track_name, 'width_phi2']  # Interpret as FWHM in degrees
        sigma_phi2 = width_phi2 / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        # Calculate HEALPix nside given the stream width such that a star at most 2 sigma away from the stream track is be guaranteed to fall into the same pixel as a track point
        nside_max = 1 / (2 * np.sqrt(3) * 2 * sigma_phi2  * (np.pi/180))
        level = max(int(np.log2(nside_max)), 1)
        nside = 2**level  # Round down to nearest power of 2

        # Convert RA/Dec (ICRS) to HEALPix pixel indices
        pixels_intersected_by_track = np.unique(hp.ang2pix(nside, stream.track.ra.deg, stream.track.dec.deg, lonlat=True, nest=True))

        # Get HEALPix pixel indices for all stars at the same level
        all_stars_pixels = (source_ids >> 35) >> (2 * (12 - level))

        # Make mask for which stars are in the footprint by doing a binary search for membership (faster and more memory efficient than np.isin)
        idx = np.searchsorted(pixels_intersected_by_track, all_stars_pixels)
        idx[idx == len(pixels_intersected_by_track)] = len(pixels_intersected_by_track) - 1
        in_footprint_mask = pixels_intersected_by_track[idx] == all_stars_pixels
        del pixels_intersected_by_track, all_stars_pixels, idx  # Free memory
        gc.collect()  # Force garbage collection

        # Make SkyCoord object for stars in footprint
        stream_stars = SkyCoord(
            ra=ra[in_footprint_mask] * u.deg,
            dec=dec[in_footprint_mask] * u.deg,
            pm_ra_cosdec=mu_ra[in_footprint_mask] * np.cos(np.deg2rad(dec[in_footprint_mask])) * u.mas / u.yr,
            pm_dec=mu_dec[in_footprint_mask] * u.mas / u.yr,
            distance=r_med_photogeo[in_footprint_mask] * u.pc,
            frame='icrs'
        )

        # Transform stars in stream footprint and stream track into stream coordinates
        stream_stars = stream_stars.transform_to(stream.stream_frame)
        stream.track = stream.track.transform_to(stream.stream_frame)

        # Extract phi1 and phi2 of stream stars
        stars_phi1 = stream_stars.phi1.to_value(u.deg)
        stars_phi2 = stream_stars.phi2.to_value(u.deg)

        # Extract phi2 and phi1 of stream track
        track_phi1 = stream.track.phi1.to_value(u.deg)
        track_phi2 = stream.track.phi2.to_value(u.deg)

        # Interpolate track phi2 vs phi1
        interp_phi2 = np.interp(stars_phi1, track_phi1, track_phi2)

        # Compute chi2 value for position perpendicular to stream
        chi2 = ((stars_phi2 - interp_phi2) / sigma_phi2)**2# - np.log(2 * np.pi) - 2 * np.log(sigma_phi2)

        del stars_phi2, track_phi2, interp_phi2 # Free memory
        gc.collect()  # Force garbage collection

        # Proper motions (if available for stream and stars)
        if mws.summary.loc[stream_track_name, 'has_pm']:
            try:
                # Widths in proper motions
                width_pm1 = mws.summary.loc[stream_track_name, 'width_pm_phi1_cosphi2']
                sigma_pm1 = width_pm1 / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
                width_pm2 = mws.summary.loc[stream_track_name, 'width_pm_phi2']
                sigma_pm2 = width_pm2 / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

                # Mask for the stars with proper motions
                pm_mask = np.isfinite(stream_stars.pm_phi1_cosphi2) & np.isfinite(stream_stars.pm_phi2)

                # Extract proper motions of stream stars
                stars_pm1 = stream_stars[pm_mask].pm_phi1_cosphi2.to_value(u.mas / u.yr)
                stars_pm2 = stream_stars[pm_mask].pm_phi2.to_value(u.mas / u.yr)

                # Extract proper motions of stream track
                track_pm1 = stream.track.pm_phi1_cosphi2.to_value(u.mas / u.yr)
                track_pm2 = stream.track.pm_phi2.to_value(u.mas / u.yr)

                # Interpolate track proper motions vs phi1
                interp_pm1 = np.interp(stars_phi1[pm_mask], track_phi1, track_pm1)
                interp_pm2 = np.interp(stars_phi1[pm_mask], track_phi1, track_pm2)

                # Compute chi2 values for proper motions
                chi2[pm_mask] += (
                    ((stars_pm1 - interp_pm1) / sigma_pm1)**2 + # - np.log(2 * np.pi) - 2 * np.log(sigma_pm1) +
                    ((stars_pm2 - interp_pm2) / sigma_pm2)**2 # - np.log(2 * np.pi) - 2 * np.log(sigma_pm2)
                )

                del pm_mask, stars_pm1, stars_pm2, track_pm1, track_pm2, interp_pm1, interp_pm2  # Free memory
                gc.collect()  # Force garbage collection
            except:
                pass

        # Distance (for if / when this becomes available)
        if False: #mws.summary.loc[stream_track_name, 'has_D']:
            try:
                # Width in distance
                width_dist = mws.summary.loc[stream_track_name, 'width_dist']  # This doesn't exist in galstreams yet
                sigma_dist = width_dist / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

                # Mask for the stars with distances
                dist_mask = np.isfinite(stream_stars.distance)

                # Extract distance of stream stars
                stars_dist = stream_stars[dist_mask].distance.to_value(u.pc)

                # Extract distance of stream track
                track_dist = stream.track.distance.to_value(u.pc)

                # Interpolate track distance vs phi1
                interp_dist = np.interp(stars_phi1[dist_mask], track_phi1, track_dist)
                
                # Compute chi2 value for distance
                chi2[dist_mask] += ((stars_dist - interp_dist) / sigma_dist)**2 # - np.log(2 * np.pi) - 2 * np.log(sigma_dist)

                del dist_mask, stars_dist, track_dist, interp_dist  # Free memory
                gc.collect()  # Force garbage collection
            except:
                pass

        # Line-of-sight velocity (for if / when this becomes available)
        if False: #mws.summary.loc[stream_track_name, 'has_vrad']:
            try:
                # Width in line-of-sight velocity
                width_vrad = mws.summary.loc[stream_track_name, 'width_vrad']  # This doesn't exist in galstreams yet
                sigma_vrad = width_vrad / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

                # Mask for the stars with line-of-sight velocities
                vrad_mask = np.isfinite(stream_stars.vrad)

                # Extract line-of-sight velocity of stream stars
                stars_vrad = stream_stars[vrad_mask].vrad.to_value(u.km / u.s)

                # Extract line-of-sight velocity of stream track
                track_vrad = stream.track.vrad.to_value(u.km / u.s)

                # Interpolate track line-of-sight velocity vs phi1
                interp_vrad = np.interp(stars_phi1[vrad_mask], track_phi1, track_vrad)

                # Compute chi2 value for line-of-sight velocity
                chi2[vrad_mask] += ((stars_vrad - interp_vrad) / sigma_vrad)**2 # - np.log(2 * np.pi) - 2 * np.log(sigma_vrad)

                del vrad_mask, stars_vrad, track_vrad, interp_vrad  # Free memory
                gc.collect()  # Force garbage collection
            except:
                pass

        del stream_stars, stars_phi1, track_phi1  # Free memory
        gc.collect()  # Force garbage collection

        # Convert chi2 to probability measure
        probs = np.exp(-0.5 * chi2)

        # Keep only stars with sigma < 2 (i.e. probability > exp(-0.5 * 2.0**2))
        prob_mask = probs > np.exp(-0.5 * 2.0**2) # / np.sqrt(2 * np.pi)  # Equivalent to being within 2 sigma of the stream in all available dimensions
        if not prob_mask.any():
            continue
        probs = probs[prob_mask]
        indices = np.where(in_footprint_mask)[0][prob_mask]
        del in_footprint_mask, chi2, prob_mask  # Free memory
        gc.collect()  # Force garbage collection

        # Assign the stream IDs and probabilities to the members
        unassigned_bool = True
        indices_stream_ids = galstreams_members_stream_ids_gdr3[indices]
        for j in range(galstreams_members_stream_ids_gdr3.shape[1]):
            # Find which members can be assigned to this column
            unassigned_indices_bool = indices_stream_ids[:, j] == max_galstream_cluster_ID + 1

            # Assign the stream IDs and probabilities to the unassigned members for this column
            unassigned_indices = indices[unassigned_indices_bool]
            galstreams_members_stream_ids_gdr3[unassigned_indices, j] = i
            galstreams_members_stream_probs_gdr3[unassigned_indices, j] = probs[unassigned_indices_bool]

            # Update the arrays to only include those that are still unassigned
            indices = indices[~unassigned_indices_bool]
            indices_stream_ids = indices_stream_ids[~unassigned_indices_bool]
            probs = probs[~unassigned_indices_bool]

            # If all members have been assigned, break
            if indices_stream_ids.shape[0] == 0:
                unassigned_bool = False
                break
        del indices_stream_ids  # Free memory
        gc.collect()  # Force garbage collection

        if unassigned_bool:
            # Add another column to the arrays
            galstreams_members_stream_ids_gdr3 = np.concatenate(
                (galstreams_members_stream_ids_gdr3, np.full((subsample_mask.size, 1), max_galstream_cluster_ID + 1, dtype=np.int64)),
                axis=1
            )
            galstreams_members_stream_probs_gdr3 = np.concatenate(
                (galstreams_members_stream_probs_gdr3, np.zeros((subsample_mask.size, 1), dtype=np.float32)),
                axis=1
            )

            # Assign the stream IDs and probabilities to the members
            galstreams_members_stream_ids_gdr3[indices, -1] = i
            galstreams_members_stream_probs_gdr3[indices, -1] = probs
    del mws, probs, indices  # Free memory
    gc.collect()  # Force garbage collection
    
    # Get the stream IDs and probabilities for the subsample in this work
    galstreams_members_stream_ids_subsample = galstreams_members_stream_ids_gdr3[subsample_mask]
    galstreams_members_stream_probs_subsample = galstreams_members_stream_probs_gdr3[subsample_mask]

    # Pre-compute the total sum of probabilities for each galstreams stream
    print("... pre-computing the total sum of probabilities for each galstreams stream")
    galstreams_streams_probability_sums_total = np.bincount(galstreams_members_stream_ids_gdr3.ravel(),
                                        weights=galstreams_members_stream_probs_gdr3.ravel(),
                                        minlength=max_galstream_cluster_ID + 1)[:max_galstream_cluster_ID + 1]  # (N_streams,)
    del galstreams_members_stream_ids_gdr3, galstreams_members_stream_probs_gdr3  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the sum of probabilities for each galstreams stream in the overlap with the subsample
    print("... pre-computing the total sum of probabilities for each galstreams stream in the subsample of this work")
    galstreams_streams_probability_sums_overlap = np.bincount(galstreams_members_stream_ids_subsample.ravel(),
                                        weights=galstreams_members_stream_probs_subsample.ravel(),
                                        minlength=max_galstream_cluster_ID + 1)[:max_galstream_cluster_ID + 1]  # (N_streams,)

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_galstreams_members_stream_ids_subsample, galstreams_members_stream_ids_subsample)
    np.save(file_path_galstreams_members_stream_probs_subsample, galstreams_members_stream_probs_subsample)
    np.save(file_path_galstreams_streams_probability_sums_total, galstreams_streams_probability_sums_total)
    np.save(file_path_galstreams_streams_probability_sums_overlap, galstreams_streams_probability_sums_overlap)
    del subsample_mask, galstreams_members_stream_ids_subsample, galstreams_members_stream_probs_subsample, galstreams_streams_probability_sums_total, galstreams_streams_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

def plot_galstreams_streams_on_sky(overwrite=False):
    """
    Plot the galstreams streams on the sky.
    """
    # Check if plot already exists
    file_streams_on_sky_path = os.path.join(FIGURES_PATH, "galstreams_streams_on_sky.png")
    if os.path.exists(file_streams_on_sky_path) and not overwrite:
        print(f"Galstreams streams on sky plot already exists at:\n\t{file_streams_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting galstreams streams on the sky...")

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians for Mollweide projection
    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)

     # Mollweide expects longitudes in the range [-pi, pi] and latitudes in the range [-pi/2, pi/2]
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1 # Invert x-axis for on-sky astro plot

    # Load the galstreams clustering output
    print("... loading galstreams clustering output for plotting")
    galstreams_members_stream_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_members_stream_ids_subsample.npy"))  # (N,)
    galstreams_members_stream_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_members_stream_probs_subsample.npy"))  # (N,)

    # Simplify the catalogue
    print("... simplifying the galstreams catalogue for plotting")
    plottable_bool = np.any(galstreams_members_stream_probs_subsample > 0.5, axis=1)
    no_stream_id = galstreams_members_stream_ids_subsample.max()
    galstreams_members_stream_ids_subsample = galstreams_members_stream_ids_subsample[plottable_bool]
    galactic_coordinates = galactic_coordinates[plottable_bool]
    print(f"... {plottable_bool.sum()} plottable stars")
    del galstreams_members_stream_probs_subsample, plottable_bool  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot streams on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the streams and plot them
    print("... plotting streams on the sky")
    for i in range(no_stream_id):
        streamMembers = np.any(galstreams_members_stream_ids_subsample == i, axis=1)
        n_stream = streamMembers.sum()
        if n_stream > 5000:
            streamMembers_indices = np.where(streamMembers)[0]
            removed_indices = np.random.choice(streamMembers_indices, size=n_stream-5000, replace=False)
            streamMembers[removed_indices] = False
        ax.scatter(
            *galactic_coordinates[streamMembers].T,
            facecolor=f"C{i}", edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each stream with a different color

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_streams_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters on sky plot to {file_streams_on_sky_path}.\n")

def compare_to_galstreams(overwrite=False):
    """
    Compare the clustering output to the galstreams catalogue.
    """
    # Check if comparison results already exist
    file_path_best_match_astrolink_clusters = os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_best_match_astrolink_clusters.npy")
    file_path_cluster_rpje = os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_rpje.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_best_match_astrolink_clusters) and
                 os.path.exists(file_path_cluster_rpje))
    if all_exist and not overwrite:
        print("Galstreams comparison results already exist at:")
        print(f"\t{file_path_best_match_astrolink_clusters} and")
        print(f"\t{file_path_cluster_rpje} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to the galstreams catalogue...")

    # Load the reduced galstreams data
    print("... loading reduced galstreams data")
    galstreams_members_stream_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_members_stream_ids_subsample.npy"))  # (N,)
    galstreams_members_stream_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_members_stream_probs_subsample.npy"))  # (N,)
    galstreams_streams_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_streams_probability_sums_total.npy"))  # (N_streams,)
    galstreams_streams_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_streams_probability_sums_overlap.npy"))  # (N_streams,)
    
    # Make the comparison
    whichClusters, RPJE = compare_to_catalogue_helper(
        galstreams_members_stream_ids_subsample,
        galstreams_members_stream_probs_subsample,
        galstreams_streams_probability_sums_total,
        galstreams_streams_probability_sums_overlap
    )
    del galstreams_members_stream_ids_subsample, galstreams_members_stream_probs_subsample, galstreams_streams_probability_sums_total, galstreams_streams_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_best_match_astrolink_clusters, whichClusters)
    np.save(file_path_cluster_rpje, RPJE)
    del whichClusters, RPJE  # Free memory
    gc.collect()  # Force garbage collection

def plot_galstreams_comparison_results(overwrite=False):
    """
    Plot the results of the comparison between clustering output and the galstreams catalogue,
    showing recovery, purity, and Jaccard index under both subsample assumptions
    ('full' and 'union'), with hatched regions indicating the bounds between them.
    """
    # Check if plot already exists
    file_path = os.path.join(FIGURES_PATH, "galstreams_comparison_results.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Galstreams comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Galstreams comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJE = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_rpje.npy"))  # (N_sigmas, N_clusters, 2, 4)
    galstreams_streams_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_streams_probability_sums_total.npy"))  # (N_streams,)
    galstreams_streams_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "galstreams/galstreams_streams_probability_sums_overlap.npy"))  # (N_streams,)
    coverage = np.zeros(galstreams_streams_probability_sums_total.shape, dtype=np.float32)  # (N_streams,)
    nonzero_mask = galstreams_streams_probability_sums_total > 0.0
    coverage[nonzero_mask] = galstreams_streams_probability_sums_overlap[nonzero_mask] / galstreams_streams_probability_sums_total[nonzero_mask]
    del galstreams_streams_probability_sums_total, galstreams_streams_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Extract per-assumption and per-statistic arrays
    R_full,  R_union  = RPJE[:, :, 0, 0], RPJE[:, :, 1, 0]
    P_full,  P_union  = RPJE[:, :, 0, 1], RPJE[:, :, 1, 1]
    J_full,  J_union  = RPJE[:, :, 0, 2], RPJE[:, :, 1, 2]
    #E_full,  E_union  = RPJE[:, :, 0, 3], RPJE[:, :, 1, 3]

    # Evidence-weighted averages
    Rbar_full  = np.sum(R_full * coverage, axis=1) / np.sum(coverage)
    Rbar_union = np.sum(R_union * coverage, axis=1) / np.sum(coverage)
    Pbar_full  = np.sum(P_full * coverage, axis=1) / np.sum(coverage)
    Pbar_union = np.sum(P_union * coverage, axis=1) / np.sum(coverage)
    Jbar_full  = np.sum(J_full * coverage, axis=1) / np.sum(coverage)
    Jbar_union = np.sum(J_union * coverage, axis=1) / np.sum(coverage)

    print("... plotting combined statistics vs significance level")

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Recovery
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_union,
            color='k', linestyle='dashed', linewidth=1.5, label='R', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full,
            color='k', linestyle='dashed', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full, Rbar_union,
                    facecolor='none', hatch='//', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Purity
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_union,
            color='k', linestyle='dotted', linewidth=1.5, label='P', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full,
            color='k', linestyle='dotted', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full, Pbar_union,
                    facecolor='none', hatch='\\', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
            color='k', linestyle='solid', linewidth=1.5, label='J', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
            color='k', linestyle='solid', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                    color='k', alpha=0.3, zorder=3)

    # Final formatting
    print("... saving figure.\n")
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Significance Level")
    ax.set_ylabel("Comparison Statistic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Compare clustering output to Vasiliev & Baumgardt (2021) ===
def prepare_Vasiliev2021_for_comparison(overwrite=False):
    """
    Prepare the Vasiliev & Baumgardt (2021) catalogue for comparison to the clustering output.
    """
    # Check if files already exist
    file_path_V21_members_cluster_ids_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_members_cluster_ids_subsample.npy")
    file_path_V21_members_cluster_probs_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_members_cluster_probs_subsample.npy")
    file_path_V21_clusters_probability_sums_total = os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_clusters_probability_sums_total.npy")
    file_path_V21_clusters_probability_sums_overlap = os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_clusters_probability_sums_overlap.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_V21_members_cluster_ids_subsample) and
                 os.path.exists(file_path_V21_members_cluster_probs_subsample) and
                 os.path.exists(file_path_V21_clusters_probability_sums_total) and
                 os.path.exists(file_path_V21_clusters_probability_sums_overlap))
    if all_exist and not overwrite:
        print("Vasiliev & Baumgardt (2021) reduced data already exists at:")
        print(f"\t{file_path_V21_members_cluster_ids_subsample} ,")
        print(f"\t{file_path_V21_members_cluster_probs_subsample} ,")
        print(f"\t{file_path_V21_clusters_probability_sums_total} , and")
        print(f"\t{file_path_V21_clusters_probability_sums_overlap} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with the Vasiliev & Baumgardt (2021) catalogue...")

    # Load required arrays
    print('... loading required arrays')
    source_ids = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_source_ids.npy"))  # (N_gdr3,)
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, 'subsample_mask.npy'))  # (N_gdr3,)

    # Load the Vasiliev & Baumgardt (2021) catalogue
    print("... loading the Vasiliev & Baumgardt (2021) catalogue")
    cat_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/clusters.zip")

    # Open the zip file and read the catalogue
    with zipfile.ZipFile(cat_path, 'r') as z:
        # Find all text files inside 'clusters/catalogues/'
        globular_cluster_files = [
            name for name in z.namelist()
            if name.startswith('catalogues/') and name.endswith('.txt')
        ]
        
        # Cycle through each cluster and save the star membership probabilities for that cluster
        max_V21_cluster_ID = len(globular_cluster_files) - 1
        V21_members_cluster_ids_gdr3 = np.full((subsample_mask.size, 1), max_V21_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_V21_cluster_ID + 1, representing no cluster
        V21_members_cluster_probs_gdr3 = np.zeros((subsample_mask.size, 1), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
        for i, name in enumerate(globular_cluster_files):
            with z.open(name) as f:
                text_stream = TextIOWrapper(f, encoding='utf-8')

                # Read until we find the header line (starts with '#')
                for line in text_stream:
                    if line.startswith('#'):
                        # Strip the '#' and whitespace, then split into column names
                        colnames = line.strip('#').strip().split()
                        break

                # Now read the remaining lines as data with those column names
                df = pd.read_csv(
                    text_stream,
                    sep=r"\s+",
                    names=colnames,
                    engine="python"  # optional: allows regex separator, ensures compatibility
                )

            # Extract only the columns of interest
            cluster_source_ids = df['source_id'].to_numpy().astype(np.int64)  # (N_cluster,)
            cluster_member_probs = df['memberprob'].to_numpy().astype(np.float32)  # (N_cluster,)
            
            # Find which stars in the full GDR3 catalogue are in this cluster
            indices = np.searchsorted(source_ids, cluster_source_ids)  # Assumes source_ids is sorted
            indices = indices[(indices < source_ids.size) & (source_ids[indices] == cluster_source_ids)]  # Keep only valid matches

            # Assign the cluster IDs and probabilities to the members
            unassigned_bool = True
            cluster_source_ids_indices = V21_members_cluster_ids_gdr3[indices]
            for j in range(V21_members_cluster_ids_gdr3.shape[1]):
                # Find which members can be assigned to this column
                unassigned_indices_bool = cluster_source_ids_indices[:, j] == max_V21_cluster_ID + 1

                # Assign the cluster IDs and probabilities to the unassigned members for this column
                unassigned_indices = indices[unassigned_indices_bool]
                V21_members_cluster_ids_gdr3[unassigned_indices, j] = i
                V21_members_cluster_probs_gdr3[unassigned_indices, j] = cluster_member_probs[unassigned_indices_bool]

                # Update the arrays to only include those that are still unassigned
                indices = indices[~unassigned_indices_bool]
                cluster_source_ids_indices = cluster_source_ids_indices[~unassigned_indices_bool]
                cluster_member_probs = cluster_member_probs[~unassigned_indices_bool]

                # If all members have been assigned, break
                if cluster_source_ids_indices.shape[0] == 0:
                    unassigned_bool = False
                    break
            del cluster_source_ids_indices, unassigned_indices_bool, unassigned_indices  # Free memory
            gc.collect()  # Force garbage collection
            
            if unassigned_bool:
                # Add another column to the arrays
                V21_members_cluster_ids_gdr3 = np.concatenate(
                    (V21_members_cluster_ids_gdr3, np.full((subsample_mask.size, 1), max_V21_cluster_ID + 1, dtype=np.int64)),
                    axis=1
                )
                V21_members_cluster_probs_gdr3 = np.concatenate(
                    (V21_members_cluster_probs_gdr3, np.zeros((subsample_mask.size, 1), dtype=np.float32)),
                    axis=1
                )

                # Assign the cluster IDs and probabilities to the members
                V21_members_cluster_ids_gdr3[indices, -1] = i
                V21_members_cluster_probs_gdr3[indices, -1] = cluster_member_probs
    del source_ids  # Free memory
    gc.collect()  # Force garbage collection
    
    # Get the cluster IDs and probabilities for the subsample in this work
    V21_members_cluster_ids_subsample = V21_members_cluster_ids_gdr3[subsample_mask]
    V21_members_cluster_probs_subsample = V21_members_cluster_probs_gdr3[subsample_mask]
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total sum of probabilities for each globular cluster in Vasiliev & Baumgardt (2021)
    print("... pre-computing the total sum of probabilities for each Vasiliev & Baumgardt (2021) globular cluster")
    V21_clusters_probability_sums_total = np.bincount(V21_members_cluster_ids_gdr3.ravel(),
                                        weights=V21_members_cluster_probs_gdr3.ravel(),
                                        minlength=max_V21_cluster_ID + 1)[:max_V21_cluster_ID + 1]  # (N_clusters,)
    del V21_members_cluster_ids_gdr3, V21_members_cluster_probs_gdr3  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the sum of probabilities for each globular cluster in the overlap with the subsample
    print("... pre-computing the total sum of probabilities for each Vasiliev & Baumgardt (2021) globular cluster in the subsample of this work")
    V21_clusters_probability_sums_overlap = np.bincount(V21_members_cluster_ids_subsample.ravel(),
                                        weights=V21_members_cluster_probs_subsample.ravel(),
                                        minlength=max_V21_cluster_ID + 1)[:max_V21_cluster_ID + 1]  # (N_clusters,)

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_V21_members_cluster_ids_subsample, V21_members_cluster_ids_subsample)
    np.save(file_path_V21_members_cluster_probs_subsample, V21_members_cluster_probs_subsample)
    np.save(file_path_V21_clusters_probability_sums_total, V21_clusters_probability_sums_total)
    np.save(file_path_V21_clusters_probability_sums_overlap, V21_clusters_probability_sums_overlap)
    del V21_members_cluster_ids_subsample, V21_members_cluster_probs_subsample, V21_clusters_probability_sums_total, V21_clusters_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

def plot_Vasiliev2021_clusters_on_sky(overwrite=False):
    """
    Plot the Vasiliev & Baumgardt (2021) globular clusters on the sky.
    """
    # Check if plot already exists
    file_clusters_on_sky_path = os.path.join(FIGURES_PATH, "Vasiliev2021_clusters_on_sky.png")
    if os.path.exists(file_clusters_on_sky_path) and not overwrite:
        print(f"Vasiliev & Baumgardt (2021) globular clusters on sky plot already exists at:\n\t{file_clusters_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Vasiliev & Baumgardt (2021) globular clusters on the sky...")

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians for Mollweide projection
    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)

     # Mollweide expects longitudes in the range [-pi, pi] and latitudes in the range [-pi/2, pi/2]
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1 # Invert x-axis for on-sky astro plot

    # Load the Vasiliev & Baumgardt (2021) clustering output
    print("... loading Vasiliev & Baumgardt (2021) clustering output for plotting")
    V21_members_cluster_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_members_cluster_ids_subsample.npy"))  # (N,)
    V21_members_cluster_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_members_cluster_probs_subsample.npy"))  # (N,)

    # Simplify the catalogue
    print("... simplifying the Vasiliev & Baumgardt (2021) catalogue for plotting")
    plottable_bool = np.any(V21_members_cluster_probs_subsample > 0.5, axis=1)
    no_cluster_id = V21_members_cluster_ids_subsample.max()
    V21_members_cluster_ids_subsample = V21_members_cluster_ids_subsample[plottable_bool]
    galactic_coordinates = galactic_coordinates[plottable_bool]
    print(f"... {plottable_bool.sum()} plottable stars")
    del V21_members_cluster_probs_subsample, plottable_bool  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot clusters on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    print("... plotting clusters on the sky")
    for i in range(no_cluster_id):
        clusterMembers = np.any(V21_members_cluster_ids_subsample == i, axis=1)
        ax.scatter(
            *galactic_coordinates[clusterMembers].T,
            facecolor=f"C{i}", edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each cluster with a different color

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_clusters_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters on sky plot to {file_clusters_on_sky_path}.\n")

def compare_to_Vasiliev2021(overwrite=False):
    """
    Compare the clustering output to the Vasiliev & Baumgardt (2021) catalogue.
    """
    # Check if comparison results already exist
    file_path_best_match_astrolink_clusters = os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev2021_best_match_astrolink_clusters.npy")
    file_path_cluster_rpje = os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev2021_rpje.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_best_match_astrolink_clusters) and
                 os.path.exists(file_path_cluster_rpje))
    if all_exist and not overwrite:
        print("Vasiliev & Baumgardt (2021) comparison results already exist at:")
        print(f"\t{file_path_best_match_astrolink_clusters} and")
        print(f"\t{file_path_cluster_rpje} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to the Vasiliev & Baumgardt (2021) catalogue...")

    # Load the reduced Vasiliev & Baumgardt (2021) data
    print("... loading reduced Vasiliev & Baumgardt (2021) data")
    V21_members_cluster_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_members_cluster_ids_subsample.npy"))  # (N,)
    V21_members_cluster_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_members_cluster_probs_subsample.npy"))  # (N,)
    V21_clusters_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_clusters_probability_sums_total.npy"))  # (N_clusters,)
    V21_clusters_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_clusters_probability_sums_overlap.npy"))  # (N_clusters,)
    
    # Make the comparison
    whichClusters, RPJE = compare_to_catalogue_helper(
        V21_members_cluster_ids_subsample,
        V21_members_cluster_probs_subsample,
        V21_clusters_probability_sums_total,
        V21_clusters_probability_sums_overlap
    )
    del V21_members_cluster_ids_subsample, V21_members_cluster_probs_subsample, V21_clusters_probability_sums_total, V21_clusters_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_best_match_astrolink_clusters, whichClusters)
    np.save(file_path_cluster_rpje, RPJE)
    del whichClusters, RPJE  # Free memory
    gc.collect()  # Force garbage collection

def plot_Vasiliev2021_comparison_results(overwrite=False):
    """
    Plot the results of the comparison between clustering output and the Vasiliev & Baumgardt (2021) catalogue,
    showing recovery, purity, and Jaccard index under both subsample assumptions
    ('full' and 'union'), with hatched regions indicating the bounds between them.
    """
    # Check if plot already exists
    file_path = os.path.join(FIGURES_PATH, "Vasiliev2021_comparison_results.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Vasiliev & Baumgardt (2021) comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Vasiliev & Baumgardt (2021) comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJE = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev2021_rpje.npy"))  # (N_sigmas, N_clusters, 2, 4)
    V21_clusters_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_clusters_probability_sums_total.npy"))  # (N_clusters,)
    V21_clusters_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Vasiliev2021/vasiliev21_clusters_probability_sums_overlap.npy"))  # (N_clusters,)
    coverage = V21_clusters_probability_sums_overlap / V21_clusters_probability_sums_total  # (N_clusters,)
    del V21_clusters_probability_sums_total, V21_clusters_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Extract per-assumption and per-statistic arrays
    R_full,  R_union  = RPJE[:, :, 0, 0], RPJE[:, :, 1, 0]
    P_full,  P_union  = RPJE[:, :, 0, 1], RPJE[:, :, 1, 1]
    J_full,  J_union  = RPJE[:, :, 0, 2], RPJE[:, :, 1, 2]
    #E_full,  E_union  = RPJE[:, :, 0, 3], RPJE[:, :, 1, 3]

    # Evidence-weighted averages
    Rbar_full  = np.sum(R_full * coverage, axis=1) / np.sum(coverage)
    Rbar_union = np.sum(R_union * coverage, axis=1) / np.sum(coverage)
    Pbar_full  = np.sum(P_full * coverage, axis=1) / np.sum(coverage)
    Pbar_union = np.sum(P_union * coverage, axis=1) / np.sum(coverage)
    Jbar_full  = np.sum(J_full * coverage, axis=1) / np.sum(coverage)
    Jbar_union = np.sum(J_union * coverage, axis=1) / np.sum(coverage)

    print("... plotting combined statistics vs significance level")

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Recovery
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_union,
            color='k', linestyle='dashed', linewidth=1.5, label='R', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full,
            color='k', linestyle='dashed', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full, Rbar_union,
                    facecolor='none', hatch='//', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Purity
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_union,
            color='k', linestyle='dotted', linewidth=1.5, label='P', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full,
            color='k', linestyle='dotted', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full, Pbar_union,
                    facecolor='none', hatch='\\', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
            color='k', linestyle='solid', linewidth=1.5, label='J', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
            color='k', linestyle='solid', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                    color='k', alpha=0.3, zorder=3)

    # Final formatting
    print("... saving figure.\n")
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Significance Level")
    ax.set_ylabel("Comparison Statistic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Compare clustering output to Battaglia et al. 2021 catalogue ===
def prepare_Battaglia2021_for_comparison(overwrite=False):
    """
    Prepare the Battaglia et al. (2021) catalogue for comparison to the clustering output.
    """
    # Check if files already exist
    file_path_dwarfgalaxies_names = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_dwarfgalaxies_names.npy")
    file_path_B21_members_dwarfgalaxy_ids_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_members_dwarfgalaxy_ids_subsample.npy")
    file_path_B21_members_dwarfgalaxy_probs_subsample = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_members_dwarfgalaxy_probs_subsample.npy")
    file_path_B21_dwarfgalaxies_probability_sums_total = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_dwarfgalaxies_probability_sums_total.npy")
    file_path_B21_dwarfgalaxies_probability_sums_overlap = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_dwarfgalaxies_probability_sums_overlap.npy")

    # Skip if all files exist and overwrite is False
    all_exist = (os.path.exists(file_path_dwarfgalaxies_names) and
                 os.path.exists(file_path_B21_members_dwarfgalaxy_ids_subsample) and
                 os.path.exists(file_path_B21_members_dwarfgalaxy_probs_subsample) and
                 os.path.exists(file_path_B21_dwarfgalaxies_probability_sums_total) and
                 os.path.exists(file_path_B21_dwarfgalaxies_probability_sums_overlap))
    if all_exist and not overwrite:
        print("Battaglia et al. (2021) reduced data already exists at:")
        print(f"\t{file_path_dwarfgalaxies_names} ,")
        print(f"\t{file_path_B21_members_dwarfgalaxy_ids_subsample} ,")
        print(f"\t{file_path_B21_members_dwarfgalaxy_probs_subsample} ,")
        print(f"\t{file_path_B21_dwarfgalaxies_probability_sums_total} , and")
        print(f"\t{file_path_B21_dwarfgalaxies_probability_sums_overlap} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with the Battaglia et al. (2021) catalogue...")

    # Read the pmem.dat.gz file
    print("... loading pmem.dat.gz data")
    readme_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/ReadMe")
    pmem_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/pmem.dat.gz")
    df_pmem = load_cds_table(readme_path, pmem_path)

    # Save the names of the dwarf galaxies
    print("... saving dwarf galaxy names")
    B21_members_dwarfgalaxies_names = df_pmem["Galaxy"].to_numpy()  # Dwarf galaxy names
    _, indices, inverse = np.unique(B21_members_dwarfgalaxies_names, return_index=True, return_inverse=True)
    B21_dwarfgalaxies_names = B21_members_dwarfgalaxies_names[np.sort(indices)]  # Unique dwarf galaxy names in the original order
    np.save(file_path_dwarfgalaxies_names, B21_dwarfgalaxies_names)
    del B21_members_dwarfgalaxies_names, _, B21_dwarfgalaxies_names  # Free memory
    gc.collect()  # Force garbage collection

    # Get the dwarf galaxy IDs, GDR3 source IDs, and membership probabilities
    print("... getting member information")
    B21_members_dwarfgalaxy_ids = np.argsort(np.argsort(indices))[inverse]  # Cluster IDs per member
    B21_members_source_ids = df_pmem['GaiaEDR3'].to_numpy()  # Source IDs of the members
    B21_members_probs = df_pmem["Pmemb"].to_numpy()  # Membership probabilities
    B21_members_probs[~np.isfinite(B21_members_probs)] = 0.0  # Set non-finite probabilities to 0
    del df_pmem, inverse, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Load subsample mask
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)

    # Save the membership mask
    print("... making membership mask")
    indices = np.searchsorted(gdr3_source_ids, B21_members_source_ids) # Assumes gdr3_source_ids is sorted
    B21_members_mask = np.zeros_like(gdr3_source_ids, dtype=np.bool_)  # Create a mask of the same shape as gdr3_source_ids
    B21_members_mask[indices] = True  # Set the indices of the members to True
    del gdr3_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Get Battaglia+2021 dwarf galaxy IDs for this subsample (multiple columns since stars can be in multiple Battaglia+2021 dwarf galaxies)
    print("... getting Battaglia+2024 dwarf galaxy IDs and membership probabilities for the subsample in this work")
    max_appearances = np.unique(B21_members_source_ids, return_counts=True)[1].max()
    max_B21_dwarfgalaxy_ID = B21_members_dwarfgalaxy_ids.max()
    B21_members_dwarfgalaxy_ids_gdr3 = np.full((B21_members_mask.size, max_appearances), max_B21_dwarfgalaxy_ID + 1, dtype=np.int64)  # (N,) Initialize with max_B21_dwarfgalaxy_ID + 1, representing no cluster
    B21_members_dwarfgalaxy_probs_gdr3 = np.zeros((B21_members_mask.size, max_appearances), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
    B21_members_mask_where = np.where(B21_members_mask)[0]  # Use indices of members in the full catalogue from now on to do efficient slicing/indexing
    del B21_members_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Assign dwarf galaxy IDs / membership probabilities to each of the members
    unique_B21_source_ids, indices, counts = np.unique(B21_members_source_ids, return_index=True, return_counts=True)
    for num_count in range(1, max_appearances + 1):
        # Get the relative position of the members in the full catalogue for this count
        num_count_bool = counts == num_count

        # Get the dwarf galaxy IDs and probabilities for the members with this count
        if num_count == 1:
            cluster_ids = B21_members_dwarfgalaxy_ids[indices[num_count_bool]][:, None]
            members_probs = B21_members_probs[indices[num_count_bool]][:, None]
        else: # There are not too many of these so what follows is efficient enough
            cluster_ids = np.zeros((num_count_bool.sum(), num_count), dtype=np.int64)
            members_probs = np.zeros((num_count_bool.sum(), num_count), dtype=np.float32)
            for i, sid in enumerate(unique_B21_source_ids[num_count_bool]):
                # Get the indices of the members with this source ID
                source_id_match = np.where(B21_members_source_ids == sid)[0]

                # Get the dwarf galaxy IDs and probabilities for these members
                cluster_ids[i] = B21_members_dwarfgalaxy_ids[source_id_match]
                members_probs[i] = B21_members_probs[source_id_match]

        # Assign the dwarf galaxy IDs and probabilities to the members
        B21_members_dwarfgalaxy_ids_gdr3[B21_members_mask_where[num_count_bool], :num_count] = cluster_ids
        B21_members_dwarfgalaxy_probs_gdr3[B21_members_mask_where[num_count_bool], :num_count] = members_probs

    B21_members_dwarfgalaxy_ids_subsample = B21_members_dwarfgalaxy_ids_gdr3[subsample_mask]
    B21_members_dwarfgalaxy_probs_subsample = B21_members_dwarfgalaxy_probs_gdr3[subsample_mask]
    del subsample_mask, B21_members_source_ids, B21_members_dwarfgalaxy_ids_gdr3, B21_members_dwarfgalaxy_probs_gdr3, B21_members_mask_where, unique_B21_source_ids, indices, counts  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total sum of probabilities for each Battaglia+2021 dwarf galaxy
    print("... pre-computing the total sum of probabilities for each Battaglia+2021 dwarf galaxy")
    B21_dwarfgalaxies_probability_sums_total = np.bincount(B21_members_dwarfgalaxy_ids.ravel(), 
                                        weights=B21_members_probs.ravel(),
                                        minlength=max_B21_dwarfgalaxy_ID + 1)  # (N_dwarfgalaxies,)
    del B21_members_dwarfgalaxy_ids, B21_members_probs  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the sum of probabilities for each Battaglia+2021 dwarf galaxy in the overlap with the subsample
    print("... pre-computing the total sum of probabilities for each Battaglia+2021 dwarf galaxy in the subsample of this work")
    B21_dwarfgalaxies_probability_sums_overlap = np.bincount(B21_members_dwarfgalaxy_ids_subsample.ravel(),
                                        weights=B21_members_dwarfgalaxy_probs_subsample.ravel(),
                                        minlength=max_B21_dwarfgalaxy_ID + 1)[:max_B21_dwarfgalaxy_ID + 1]  # (N_dwarfgalaxies,)

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_B21_members_dwarfgalaxy_ids_subsample, B21_members_dwarfgalaxy_ids_subsample)
    np.save(file_path_B21_members_dwarfgalaxy_probs_subsample, B21_members_dwarfgalaxy_probs_subsample)
    np.save(file_path_B21_dwarfgalaxies_probability_sums_total, B21_dwarfgalaxies_probability_sums_total)
    np.save(file_path_B21_dwarfgalaxies_probability_sums_overlap, B21_dwarfgalaxies_probability_sums_overlap)
    del B21_members_dwarfgalaxy_ids_subsample, B21_members_dwarfgalaxy_probs_subsample, B21_dwarfgalaxies_probability_sums_total, B21_dwarfgalaxies_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

def plot_Battaglia2021_dwarfgalaxies_on_sky(overwrite=False):
    """
    Plot the Battaglia et al. (2021) dwarf galaxies on the sky.
    """
    # Check if plot already exists
    file_clusters_on_sky_path = os.path.join(FIGURES_PATH, "Battaglia2021_dwarfgalaxies_on_sky.png")
    if os.path.exists(file_clusters_on_sky_path) and not overwrite:
        print(f"Battaglia et al. (2021) dwarf galaxies on sky plot already exists at:\n\t{file_clusters_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Battaglia et al. (2021) dwarf galaxies on the sky...")

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians for Mollweide projection
    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)

     # Mollweide expects longitudes in the range [-pi, pi] and latitudes in the range [-pi/2, pi/2]
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1 # Invert x-axis for on-sky astro plot

    # Load the Battaglia et al. (2021) clustering output
    print("... loading Battaglia et al. (2021) clustering output for plotting")
    B21_members_dwarfgalaxy_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_members_dwarfgalaxy_ids_subsample.npy"))  # (N,)
    B21_members_dwarfgalaxy_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_members_dwarfgalaxy_probs_subsample.npy"))  # (N,)

    # Simplify the catalogue
    print("... simplifying the Battaglia et al. (2021) catalogue for plotting")
    plottable_bool = np.any(B21_members_dwarfgalaxy_probs_subsample > 0.5, axis=1)
    no_dwarfgalaxy_id = B21_members_dwarfgalaxy_ids_subsample.max()
    B21_members_dwarfgalaxy_ids_subsample = B21_members_dwarfgalaxy_ids_subsample[plottable_bool]
    galactic_coordinates = galactic_coordinates[plottable_bool]
    print(f"... {plottable_bool.sum()} plottable stars")
    del B21_members_dwarfgalaxy_probs_subsample, plottable_bool  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot dwarf galaxies on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the dwarf galaxies and plot them
    print("... plotting dwarf galaxies on the sky")
    for i in range(no_dwarfgalaxy_id):
        dwarfgalaxyMembers = np.any(B21_members_dwarfgalaxy_ids_subsample == i, axis=1)
        ax.scatter(
            *galactic_coordinates[dwarfgalaxyMembers].T,
            facecolor=f"C{i}", edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each dwarf galaxy with a different color

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_clusters_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved dwarf galaxies on sky plot to {file_clusters_on_sky_path}.\n")

def compare_to_Battaglia2021(overwrite=False):
    """
    Compare the clustering output to the Battaglia et al. (2021) catalogue.
    """
    # Check if comparison results already exist
    file_path_best_match_astrolink_clusters = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_best_match_astrolink_clusters.npy")
    file_path_cluster_rpje = os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_rpje.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_cluster_rpje) and
                 os.path.exists(file_path_best_match_astrolink_clusters))
    if all_exist and not overwrite:
        print("Battaglia et al. (2021) comparison results already exist at:")
        print(f"\t{file_path_best_match_astrolink_clusters} and")
        print(f"\t{file_path_cluster_rpje} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to Battaglia et al. (2021)...")

    # Load required arrays
    print("... loading required arrays for comparison")
    B21_members_dwarfgalaxy_ids_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_members_dwarfgalaxy_ids_subsample.npy"))  # (N,)
    B21_members_dwarfgalaxy_probs_subsample = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_members_dwarfgalaxy_probs_subsample.npy"))  # (N,)
    B21_dwarfgalaxies_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_dwarfgalaxies_probability_sums_total.npy"))  # (N_clusters,)
    B21_dwarfgalaxies_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_dwarfgalaxies_probability_sums_overlap.npy"))  # (N_clusters,)

    # Make the comparison
    whichClusters, RPJE = compare_to_catalogue_helper(
        B21_members_dwarfgalaxy_ids_subsample,
        B21_members_dwarfgalaxy_probs_subsample,
        B21_dwarfgalaxies_probability_sums_total,
        B21_dwarfgalaxies_probability_sums_overlap
    )
    del B21_members_dwarfgalaxy_ids_subsample, B21_members_dwarfgalaxy_probs_subsample, B21_dwarfgalaxies_probability_sums_total, B21_dwarfgalaxies_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_best_match_astrolink_clusters, whichClusters)
    np.save(file_path_cluster_rpje, RPJE)
    del whichClusters, RPJE  # Free memory
    gc.collect()  # Force garbage collection

def plot_Battaglia2021_comparison_results(overwrite=False):
    """
    Plot the results of the comparison between clustering output and the Battaglia et al. (2021) catalogue,
    showing recovery, purity, and Jaccard index under both subsample assumptions
    ('full' and 'union'), with hatched regions indicating the bounds between them.
    """
    # Check if plot already exists
    file_path = os.path.join(FIGURES_PATH, "Battaglia2021_comparison_results.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Battaglia et al. (2021) comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Battaglia et al. (2021) comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJE = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_rpje.npy"))  # (N_sigmas, N_dwarfgalaxies, 2, 4)
    B21_dwarfgalaxies_probability_sums_total = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_dwarfgalaxies_probability_sums_total.npy"))  # (N_dwarfgalaxies,)
    B21_dwarfgalaxies_probability_sums_overlap = np.load(os.path.join(AUXILLARY_CATALOGUES_PATH, "Battaglia2021/battaglia21_dwarfgalaxies_probability_sums_overlap.npy"))  # (N_dwarfgalaxies,)
    coverage = B21_dwarfgalaxies_probability_sums_overlap / B21_dwarfgalaxies_probability_sums_total  # (N_dwarfgalaxies,)
    del B21_dwarfgalaxies_probability_sums_total, B21_dwarfgalaxies_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Extract per-assumption and per-statistic arrays
    R_full,  R_union  = RPJE[:, :, 0, 0], RPJE[:, :, 1, 0]
    P_full,  P_union  = RPJE[:, :, 0, 1], RPJE[:, :, 1, 1]
    J_full,  J_union  = RPJE[:, :, 0, 2], RPJE[:, :, 1, 2]
    #E_full,  E_union  = RPJE[:, :, 0, 3], RPJE[:, :, 1, 3]

    # Evidence-weighted averages
    Rbar_full  = np.sum(R_full * coverage, axis=1) / np.sum(coverage)
    Rbar_union = np.sum(R_union * coverage, axis=1) / np.sum(coverage)
    Pbar_full  = np.sum(P_full * coverage, axis=1) / np.sum(coverage)
    Pbar_union = np.sum(P_union * coverage, axis=1) / np.sum(coverage)
    Jbar_full  = np.sum(J_full * coverage, axis=1) / np.sum(coverage)
    Jbar_union = np.sum(J_union * coverage, axis=1) / np.sum(coverage)

    print("... plotting combined statistics vs significance level")

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Recovery
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_union,
            color='k', linestyle='dashed', linewidth=1.5, label='R', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full,
            color='k', linestyle='dashed', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Rbar_full, Rbar_union,
                    facecolor='none', hatch='//', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Purity
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_union,
            color='k', linestyle='dotted', linewidth=1.5, label='P', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full,
            color='k', linestyle='dotted', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full, Pbar_union,
                    facecolor='none', hatch='\\', edgecolor='k', linewidth=0.0, alpha=0.3, zorder=3)

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
            color='k', linestyle='solid', linewidth=1.5, label='J', zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
            color='k', linestyle='solid', linewidth=1.5, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                    color='k', alpha=0.3, zorder=3)

    # Final formatting
    print("... saving figure.\n")
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Significance Level")
    ax.set_ylabel("Comparison Statistic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Run script ===
if __name__ == "__main__":
    # Ensure paths exist
    os.makedirs(REDUCED_CATALOGUE_PATH, exist_ok=True)
    os.makedirs(SUBSAMPLE_PATH, exist_ok=True)
    os.makedirs(CLUSTERING_PATH, exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Reduce raw catalogue files to numpy files
    prepare_gdr3_catalogue()
    prepare_bailerjones_gedr3_distances()

    # Calculate empirical selection function
    calculate_empirical_survey_selection_function()
    plot_limiting_g_band_magnitude_on_sky()
    
    # Construct subsample and subsample selection function
    construct_subsample_from_full_catalogue()
    calculate_subsample_selection_function()

    # Calculate total selection function for subsample
    calculate_total_selection_function_for_subsample(True)
    plot_total_selection_function_for_subsample(True)
    
    # Construct input data to be passed to AstroLink
    calculate_distance_contraction_for_subsample(True)
    calculate_contracted_data_and_errors_for_subsample(True)
    construct_data_space_for_subsample(True)

    # Apply AstroLink to subsample and plot of cluster properties
    apply_astrolink_to_subsample(True)
    plot_prominence_model_fit(True)
    plot_cluster_labels_on_sky(True)
    plot_cluster_proper_motions_on_sky(True)

    # Compare to Hunt & Reffert (2024)
    prepare_Hunt2024_for_comparison()
    #plot_Hunt2024_clusters_on_sky()
    compare_to_Hunt2024(True)
    plot_Hunt2024_comparison_results(True)

    # Compare to Unified Cluster Catalogue
    prepare_UCC_for_comparison()
    #plot_UCC_clusters_on_sky()
    compare_to_UCC(True)
    plot_UCC_comparison_results(True)

    # Compare to galstreams catalogue
    prepare_galstreams_for_comparison()
    #plot_galstreams_streams_on_sky()
    compare_to_galstreams(True)
    plot_galstreams_comparison_results(True)

    # Compare to Vasiliev & Baumgardt (2021)
    prepare_Vasiliev2021_for_comparison()
    #plot_Vasiliev2021_clusters_on_sky()
    compare_to_Vasiliev2021(True)
    plot_Vasiliev2021_comparison_results(True)

    # Compare to Battaglia et al. (2021)
    prepare_Battaglia2021_for_comparison()
    #plot_Battaglia2021_dwarfgalaxies_on_sky()
    compare_to_Battaglia2021(True)
    plot_Battaglia2021_comparison_results(True)