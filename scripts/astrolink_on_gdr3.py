# Standard imports
import os
import sys

# Parallelisation setup, restarts the script with a fresh interpreter state and forces the number of threads to be used.
# (this might not be necessary in most cases, but is included for full control in case of a misbehaving environment)
MAX_PARALLEL_WORKERS = min(os.cpu_count(), 48)  # Use up to 48 workers or all available CPUs, whichever is smaller
if "THREAD_CONTROL_INIT" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = f"{MAX_PARALLEL_WORKERS}"
    os.environ["NUMBA_NUM_THREADS"] = f"{MAX_PARALLEL_WORKERS}"
    os.environ["NUMBA_DEFAULT_NUM_THREADS"] = f"{MAX_PARALLEL_WORKERS}"
    os.environ["THREAD_CONTROL_INIT"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from numba import njit, set_num_threads
set_num_threads(MAX_PARALLEL_WORKERS) # For some reason this is necessary to get both numba AND pykdtree to use the correct number of threads

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

# Third-party numerical imports
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, beta
from scipy.special import gamma
from pykdtree.kdtree import KDTree

# Astro-specific imports
from astropy.table import Table # Works using v6.1.2, but v7.1.0 seems to try and convert 'null' values to float before using fill_values
from astropy.coordinates import SkyCoord
from astropy import units
from gaiaunlimited.selectionfunctions import m10_to_completeness
import galstreams # Also seems to need astropy==6.1.2 as also(?) gala==1.9.1

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import healpy as hp
from healpy.newvisufunc import projview

# AstroLink imports
from astrolink import AstroLink
from astrolink.io import loadAstroLinkObject, saveAstroLinkObject


# === Define script configuration ===
# User-defined paths
GDR3_CATALOGUE_PATH = "/home/_data/Gaia/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"  # Path to raw gdr3 catalogue files
AUXILLARY_CATALOGUES_PATH = "/home/williamoliver_data/gaia_clustering/auxillary_catalogues/"  # Path to auxillary catalogues (e.g. Bailer-Jones GEDR3 distances, Hunt+2024 open clusters)
WORKING_DIRECTORY = "/home/williamoliver_data/gaia_clustering/"  # Path to output files

# Relative paths
INTERMEDIATE_FILES_PATH = os.path.join(WORKING_DIRECTORY, "intermediate_files_6D/")  # Path to intermediary numpy files
OUTPUT_PATH = os.path.join(WORKING_DIRECTORY, "results_6D_rhalf_50_PMmetric-4_VRmetric-4/")  # Path to AstroLink results

# Working memory for k-nearest-neighbour retrieval
WORKING_MEMORY = 200 * (2**30)  # 200 GB (in bytes) for max memory usage by kNN queries

# Pipeline setup
WITH_RADIAL_VELOCITIES = False  # Whether to use radial velocities in the input data space for AstroLink clustering
STOCHASTIC_RUN = False  # Whether to sample stochastic values from their distributions

# Subsample construction parameters
KNN_FOR_SELECTION_FUNCTION = 128  # Number of nearest neighbors for selection function calculations
SURVEY_SF_LOWER_LIMIT = 0.99  # Empirical survey selection function lower limit for subsample stars
RUWE_UPPER_LIMIT = 1.2  # RUWE threshold for subsample stars

# Data space construction parameters
R_HALF_PERCENTILE = 50.0 + STOCHASTIC_RUN * np.random.uniform(-25, 25, 1)[0]  # Distances are contracted according to R_HALF * np.arctan(distance / R_HALF), R_HALF (in pc) marks the half-way point between full and zero Cartesian influence of the distance estimate on the clustering output
RELATIVE_PROPER_MOTION_RESCALE_FACTOR = 2.0**(1 + STOCHASTIC_RUN * np.random.normal(0.0, 0.5, 1)[0])  # An additional rescale factor for the influence of velocities relative to positions in the data space metric
RELATIVE_RADIAL_VELOCITY_RESCALE_FACTOR = 2.0**(-4 + STOCHASTIC_RUN * np.random.normal(0.0, 0.5, 1)[0])  # An additional rescale factor for the influence of radial velocities relative to positions in the data space metric (only relevant if WITH_RADIAL_VELOCITIES=True)

# AstroLink parameters
KNN_FOR_ASTROLINK = 16  # Number of nearest neighbors that AstroLink uses to calculate local densities
OPTIMAL_SIGMA_THRESHOLD = 3.5 + STOCHASTIC_RUN * np.random.normal(0, 0.1, 1)[0]  # Optimal significance threshold determined from prominence model fitting

# Comparison parameters
SIGMA_THRESHOLDS_FOR_COMPARISONS = np.linspace(2, 10, 81)  # Significance levels from 2 to 10 to be used when comparing to existing cluster catalogues
if OPTIMAL_SIGMA_THRESHOLD not in SIGMA_THRESHOLDS_FOR_COMPARISONS:
    SIGMA_THRESHOLDS_FOR_COMPARISONS = np.sort(np.append(SIGMA_THRESHOLDS_FOR_COMPARISONS, OPTIMAL_SIGMA_THRESHOLD))


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
        'radial_velocities': ['radial_velocity'],
        'astrometric_errors': ['ra_error', 'dec_error', 'pmra_error', 'pmdec_error'],
        'radial_velocity_errors': ['radial_velocity_error'],
        'astrometric_matched_transits': ['astrometric_matched_transits'],
        'photometry': ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
        'photometric_snr': ['phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error', 'phot_rp_mean_flux_over_error'],
        'ruwe': ['ruwe']
    }

    file_paths = [
        os.path.join(INTERMEDIATE_FILES_PATH, f'gdr3_{group_name}.npy')
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
        group_files = sorted(glob(os.path.join(INTERMEDIATE_FILES_PATH, f'gdr3_{group_name}_*.npy')))
        arrays = [np.load(f) for f in group_files]
        combined = np.concatenate(arrays).squeeze()

        final_path = os.path.join(INTERMEDIATE_FILES_PATH, f'gdr3_{group_name}.npy')
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
        os.path.exists(os.path.join(INTERMEDIATE_FILES_PATH, f'gdr3_{group_name}_{chunk_name}.npy'))
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
        file_path = os.path.join(INTERMEDIATE_FILES_PATH, f'gdr3_{group_name}_{chunk_name}.npy')
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
        os.path.join(INTERMEDIATE_FILES_PATH, f'bailerjones_{group_name}.npy')
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
            os.path.exists(os.path.join(INTERMEDIATE_FILES_PATH, f'bailerjones_{group_name}_{i}.npy'))
            for group_name in column_groups
        )
        if all_exist:
            continue
        
        # Convert and save each group
        for group_name, group_cols in column_groups.items():
            columns_data = [np.array(chunk[col]) for col in group_cols]  # each is 1D array of length n
            array = np.column_stack(columns_data).squeeze()
            file_path = os.path.join(INTERMEDIATE_FILES_PATH, f'bailerjones_{group_name}_{i}.npy')
            np.save(file_path, array)
        
        del chunk, columns_data, array, file_path  # Free memory
        gc.collect()

    # Merge all intermediate .npy files by group
    print("... merging temporary numpy files into final arrays (indexed with respect to the gdr3 catalogue)")
    for i, group_name in enumerate(column_groups.keys()):
        group_files = sorted(glob(os.path.join(INTERMEDIATE_FILES_PATH, f'bailerjones_{group_name}_*.npy')))
        arrays = [np.load(f) for f in group_files]
        combined = np.concatenate(arrays)

        if i == 0:  # source_id is the first group, so we can use it to re-index
            # Load GDR3 source IDs and Bailer-Jones source IDs
            print("... loading GDR3 and Bailer-Jones source IDs")
            gdr3_source_ids = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_source_ids.npy"))  # (n,)
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
            final_path = os.path.join(INTERMEDIATE_FILES_PATH, f'bailerjones_{group_name}.npy')
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
    file_path_m10_stars = os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_selection_function_m10_stars.npy")
    file_path_sf = os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_selection_function.npy")
    file_path_m10_healpix = os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_selection_function_m10_healpix.npy")
    if os.path.exists(file_path_m10_stars) and os.path.exists(file_path_sf) and os.path.exists(file_path_m10_healpix) and not overwrite:
        print(f"Empirical selection function and m10 values for the centre of HEALPix pixels already exist at:")
        print(f"\t{file_path_sf} and")
        print(f"\t{file_path_m10_healpix} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating empirical survey selection function...")

    # Load required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))  # shape (n, 2)
    G_band_magnitudes = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_photometry.npy"))[:, 0]           # shape (n,)
    astrometric_matched_transits = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_astrometric_matched_transits.npy")) # shape (n,)

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
    chunk_n_rows = min(int(WORKING_MEMORY // (16 * KNN_FOR_SELECTION_FUNCTION)), n)

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
    nside = 2**12
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
    chunk_n_rows = min(int(WORKING_MEMORY // (16 * KNN_FOR_SELECTION_FUNCTION)), npix)

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
    file_m10_path = os.path.join(OUTPUT_PATH, "gdr3_selection_function_m10_map.png")
    file_limiting_g_mag_path = os.path.join(OUTPUT_PATH, "gdr3_selection_function_limiting_g_mag.png")
    if os.path.exists(file_m10_path) and os.path.exists(file_limiting_g_mag_path) and not overwrite:
        print(f"Plots of m10 map and limiting G-band magnitude already exist at:")
        print(f"\t{file_m10_path} and")
        print(f"\t{file_limiting_g_mag_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting M10 map across the sky...")

    # Load m10 values for HEALPix pixels
    m10 = np.load(f"{INTERMEDIATE_FILES_PATH}/gdr3_selection_function_m10_healpix.npy")  # (npix,)

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
    print(f"... saved mollview plot to {file_m10_path}.\n")


    print("Plotting limiting G-band magnitude across the sky...")
    # Taken from the source code of gaiaunlimited.selectionfunctions.m10_to_completeness...
    # These are the best-fit value of the free parameters we optimised in their model:
    ax=0.9848761394197864
    bx=0.6473155510230146
    cx=0.6929084598209412
    ay=-0.003935382139847386
    by=0.2230529402297744
    cy=-0.09331877468160235
    az=0.006144107896473064
    bz=0.03681705933744438
    cz=0.35140564525722895
    lim=20.519369625540833

    # Calculate predicted parameters based on m10
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
    mask_path = os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy")
    if os.path.exists(mask_path) and not overwrite:
        print(f"Subsample mask already exists at:\n\t{mask_path} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Constructing subsample from full catalogue...")

    # Load selection function and galactic coordinates
    print("... loading required arrays from reduced catalogue")
    selection_function = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_selection_function.npy"))  # (n,)
    galactic_coords = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))  # (n, 2) in degrees
    r_med_photogeo = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "bailerjones_r_med_photogeo.npy"))  # (n,)
    proper_motions = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_proper_motions.npy"))  # (n, 2) in mas/yr
    ruwe = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_ruwe.npy"))  # (n,)
    g_mag = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_photometry.npy"))[:, 0]  # (n,)
    if WITH_RADIAL_VELOCITIES:
        radial_velocities = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_radial_velocities.npy"))  # (n,) in km/s

    # Create boolean mask for S_Gaia > threshold, ruwe < threshold, and valid astrometric data
    print("... creating subsample mask with selection function and data quality cuts")
    subsample_mask = selection_function > SURVEY_SF_LOWER_LIMIT
    subsample_mask &= ruwe < RUWE_UPPER_LIMIT
    subsample_mask &= np.isfinite(r_med_photogeo)
    subsample_mask &= np.isfinite(proper_motions).all(axis=1)
    subsample_mask &= np.isfinite(g_mag)  # Valid G-band magnitude
    if WITH_RADIAL_VELOCITIES:
        subsample_mask &= np.isfinite(radial_velocities)  # Valid radial velocity

    # Save mask
    np.save(mask_path, subsample_mask)
    print(f"... saved subsample mask to {mask_path} (selected {subsample_mask.sum()} stars).\n")

def calculate_subsample_selection_function(overwrite=False):
    """
    Calculate the subsample selection function using kNN-based metric.
    """
    # Check if total selection function already exists
    file_subsample_sf_stars = os.path.join(INTERMEDIATE_FILES_PATH, "subsample_selection_function.npy")
    if os.path.exists(file_subsample_sf_stars) and not overwrite:
        print(f"Subsample selection function already exists at:\n\t{file_subsample_sf_stars} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating subsample selection function...")

    # Load required arrays
    print("... loading required arrays")
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))  # shape (n, 2)
    G_band_magnitudes = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_photometry.npy"))[:, 0]  # shape (n,)
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # shape (n,)

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
    chunk_n_rows = min(int(WORKING_MEMORY // (16 * KNN_FOR_SELECTION_FUNCTION)), n)

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
def calculate_total_selection_function(overwrite=False):
    """
    Calculate the total selection function for the subsample.
    """
    # Check if arrays already exists
    file_nsub = os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nsub.npy")
    file_nmw = os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nmw.npy")
    file_nsub_healpix = os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nsub_healpix.npy")
    file_nmw_healpix = os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nmw_healpix.npy")

    # Skip processing if all output files already exist
    all_exist = (os.path.exists(file_nsub) and
                os.path.exists(file_nmw) and
                os.path.exists(file_nsub_healpix) and
                os.path.exists(file_nmw_healpix))
    if all_exist and not overwrite:
        print(f"Total selection function arrays already exist at:")
        print(f"\t{file_nsub} ,")
        print(f"\t{file_nmw} ,")
        print(f"\t{file_nsub_healpix} , and")
        print(f"\t{file_nmw_healpix} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating total selection function for the subsample...")

    # Load the required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))
    G_band_magnitudes = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_photometry.npy"))[:, 0]
    survey_sf = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_selection_function.npy"))
    subsample_sf = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_selection_function.npy"))

    # Identify stars with valid G magnitude
    print("... identifying valid G-band magnitudes")
    valid_gmag = np.isfinite(G_band_magnitudes)
    del G_band_magnitudes  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate the inverse of the empirical survey selection function for the subsample
    inverse_survey_sf = 1 / np.sqrt(survey_sf[valid_gmag]**2 + 1 / KNN_FOR_SELECTION_FUNCTION**2)  # Soft floor to avoid diverging values
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
    chunk_n_rows = min(int(WORKING_MEMORY // (16 * KNN_FOR_SELECTION_FUNCTION)), n)

    # Initialize total selection function array for stars in the subsample
    print("... initializing total selection function arrays for stars in the subsample")
    nsub = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    nmw = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    valid_gmag = np.where(valid_gmag)[0]  # Indices of stars with valid G-band magnitudes

    # Compute total selection function for each star in the subsample
    for start in range(0, n, chunk_n_rows):
        end = min(start + chunk_n_rows, n)

        print(f"... computing total selection function for each star in subsample -- batch {start // chunk_n_rows + 1} of {n // chunk_n_rows + 1}   ", end='\r')
        # k-nearest neighbours query
        _, idx = tree.query(
            xyz_stars[start:end],
            k=KNN_FOR_SELECTION_FUNCTION,
            sqr_dists=True
        )
        del _  # Free memory
        gc.collect()  # Force garbage collection

        # Compute expected number of neighbours in subsample and in Milky Way
        valid_slice = valid_gmag[start:end]
        nsub[valid_slice] = subsample_sf[idx].sum(axis=1)
        nmw[valid_slice] = inverse_survey_sf[idx].sum(axis=1)

        # Delete temporary variables to free memory
        del idx
        gc.collect()
    print(f"... range of expected number of neighbours in subsample: {np.nanmin(nsub):.3f} -- {np.nanmax(nsub):.3f}                   ")
    print(f"... range of expected number of neighbours in Milky Way: {np.nanmin(nmw):.3f} -- {np.nanmax(nmw):.3f}")

    # Save total selection function arrays for stars
    print(f"... saving total selection function arrays for stars to:")
    print(f"\t{file_nsub} and")
    print(f"\t{file_nmw} .\n")
    np.save(file_nsub, nsub)
    np.save(file_nmw, nmw)

    del xyz_stars, nsub, nmw  # Free memory
    gc.collect()  # Force garbage collection

    # Also calculate the total selection function values at the centre of each HEALPix pixel for plotting
    print("Calculating total selection function for HEALPix pixels...")
    nside = 2**12
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
    chunk_n_rows = min(int(WORKING_MEMORY // (16 * KNN_FOR_SELECTION_FUNCTION)), npix)

    # Initialize arrays for HEALPix pixels
    print("... initializing total selection function arrays for HEALPix pixels")
    nsub_healpix = np.empty(npix)
    nmw_healpix = np.empty(npix)

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
        del _ # Free memory
        gc.collect()  # Force garbage collection

        # Total selection function is the posterior distribution Beta(n_sub + 1, n_mw - n_sub + 1)
        nsub_healpix[start:end] = subsample_sf[idx].sum(axis=1)
        nmw_healpix[start:end] = inverse_survey_sf[idx].sum(axis=1)

        # Delete temporary variables to free memory
        del idx
        gc.collect()
    
    del tree, xyz_healpix, valid_gmag, subsample_sf, inverse_survey_sf  # Free memory
    gc.collect()  # Force garbage collection

    # Save total selection function arrays for healpix pixels
    print(f"... saving total selection function arrays for HEALPix pixels to:                ")
    print(f"\t{file_nsub_healpix} and")
    print(f"\t{file_nmw_healpix} .\n")
    np.save(file_nsub_healpix, nsub_healpix)
    np.save(file_nmw_healpix, nmw_healpix)
    del nsub_healpix, nmw_healpix  # Free memory
    gc.collect()  # Force garbage collection

def plot_total_selection_function(overwrite=False):
    """
    Plot the mean and standard error of the total selection function across the sky using HEALPix.
    """
    # Check if plots already exist
    file_total_sf_mean_path = os.path.join(OUTPUT_PATH, "total_selection_function_mean.png")
    file_total_sf_se_path = os.path.join(OUTPUT_PATH, "total_selection_function_stderr.png")
    file_total_sf_se_over_mean_path = os.path.join(OUTPUT_PATH, "total_selection_function_stderr_over_mean.png")
    if os.path.exists(file_total_sf_mean_path) and os.path.exists(file_total_sf_se_path) and not overwrite:
        print(f"Total selection function mean and standard error plots already exist at:")
        print(f"\t{file_total_sf_mean_path} ,")
        print(f"\t{file_total_sf_se_path} , and")
        print(f"\t{file_total_sf_se_over_mean_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting total selection function on the sky...")

    # Load total selection function data
    nsub_healpix = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nsub_healpix.npy"))  # (npix,)
    nmw_healpix = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nmw_healpix.npy"))  # (npix,)

    # Compute mean and standard error of the total selection function
    print("... computing mean and standard error of the total selection function")
    total_sf_mean = (nsub_healpix + 1) / (nmw_healpix + 2)
    total_sf_se = np.sqrt(total_sf_mean * (nmw_healpix - nsub_healpix + 1) / ((nmw_healpix + 2) * nsub_healpix + 3))

    # Create a Mollweide projection plot of the total selection function mean
    plt.figure(figsize=(12, 6))

    projview(
        total_sf_mean,
        coord=["G"],
        nest=True,
        unit=r"Total selection function mean, $\mu_{S_\mathrm{total}}$",
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
        unit=r"Total selection function standard error, $\sigma_{S_\mathrm{total}}$",
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
        unit=r"Total selection function fractional error, $\sigma_{S_\mathrm{total}} / \mu_{S_\mathrm{total}}$",
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
def calculate_contracted_subspaces_and_errors(overwrite=False):
    """
    Calculate non-linear contractions for Cartesian positions and velocities and their uncertainties.
    """
    # Check if contraction arrays already exists
    file_path_contracted_positions = os.path.join(INTERMEDIATE_FILES_PATH, "contracted_positions.npy")
    file_path_contracted_position_uncertainties = os.path.join(INTERMEDIATE_FILES_PATH, "contracted_position_uncertainties.npy")
    file_path_contracted_velocities = os.path.join(INTERMEDIATE_FILES_PATH, "contracted_velocities.npy")
    file_path_contracted_velocity_uncertainties = os.path.join(INTERMEDIATE_FILES_PATH, "contracted_velocity_uncertainties.npy")
    file_path_radial_velocities_3D = os.path.join(INTERMEDIATE_FILES_PATH, "radial_velocities_3D.npy")
    file_path_radial_velocity_uncertainties_3D = os.path.join(INTERMEDIATE_FILES_PATH, "radial_velocity_uncertainties_3D.npy")
    
    if WITH_RADIAL_VELOCITIES:
        # Skip processing if all output files already exist
        all_exist = (
            os.path.exists(file_path_contracted_positions) and
            os.path.exists(file_path_contracted_position_uncertainties) and
            os.path.exists(file_path_contracted_velocities) and
            os.path.exists(file_path_contracted_velocity_uncertainties) and
            os.path.exists(file_path_radial_velocities_3D) and
            os.path.exists(file_path_radial_velocity_uncertainties_3D)
        )
        if all_exist and not overwrite:
            print(f"Subspace contraction arrays already exist at:")
            print(f"\t{file_path_contracted_positions} ,")
            print(f"\t{file_path_contracted_position_uncertainties} ,")
            print(f"\t{file_path_contracted_velocities} ,")
            print(f"\t{file_path_contracted_velocity_uncertainties} ,")
            print(f"\t{file_path_radial_velocities_3D} , and")
            print(f"\t{file_path_radial_velocity_uncertainties_3D} .")
            print("Use overwrite=True to force recomputation.\n")
            return
    else:
        # Skip processing if all output files already exist
        all_exist = (
            os.path.exists(file_path_contracted_positions) and
            os.path.exists(file_path_contracted_position_uncertainties) and
            os.path.exists(file_path_contracted_velocities) and
            os.path.exists(file_path_contracted_velocity_uncertainties)
        )
        if all_exist and not overwrite:
            print(f"Subspace contraction arrays already exist at:")
            print(f"\t{file_path_contracted_positions} ,")
            print(f"\t{file_path_contracted_position_uncertainties} ,")
            print(f"\t{file_path_contracted_velocities} , and")
            print(f"\t{file_path_contracted_velocity_uncertainties} .")
            print("Use overwrite=True to force recomputation.\n")
            return
    print("Calculating subspace contractions for subsample...")

    # Load required arrays
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # shape (N,)

    print('... loading astrometric solution for subsample')
    r = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "bailerjones_r_med_photogeo.npy"))[subsample_mask]  # shape (N,) in pc
    ra, dec = np.deg2rad(np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_equatorial_coordinates.npy"))[subsample_mask]).T  # shape (N, 2) in radians
    cos_ra, sin_ra, cos_dec, sin_dec = np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec) # Trigonemetric components
    mu_ra, mu_dec = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_proper_motions.npy"))[subsample_mask].T  # shape (N, 2) in mas/yr
    del ra, dec  # Free memory
    gc.collect()  # Force garbage collection

    print('... loading astrometric solution uncertainties for subsample')
    lo, high = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "bailerjones_r_lo_high_photogeo.npy"))[subsample_mask].T  # shape (N, 2) in pc
    delta_r = (high - lo) / 2  # Symmetrize the distance uncertainty for first-order propagation
    delta_ra, delta_dec, delta_mu_ra, delta_mu_dec = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_astrometric_errors.npy"))[subsample_mask].T  # shape (N, 4) in [mas, mas, mas/yr, mas/yr]
    delta_ra, delta_dec = delta_ra * (np.pi / 180 / 3600000), delta_dec * (np.pi / 180 / 3600000)  # Convert angular errors from mas to radians
    del lo, high  # Free memory
    gc.collect()  # Force garbage collection

    if WITH_RADIAL_VELOCITIES:
        print('... loading radial velocities and uncertainties for subsample')
        vr = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_radial_velocities.npy"))[subsample_mask]  # shape (N,) in km/s
        delta_vr = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_radial_velocity_errors.npy"))[subsample_mask]  # shape (N,) in km/s
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate r_half
    r_half = np.percentile(r / np.sqrt(delta_r + 1e-6), R_HALF_PERCENTILE)  # Use quantiles as a more robust estimator
    print(f"... calculated r_half = {r_half:.3f} pc")
    
    # Calculate contracted distances and their uncertainties
    fr = r_half * np.arctan(r / r_half)  # shape (N,)
    dfr_dr = r_half**2 / (r_half**2 + r**2)  # shape (N,)
    delta_fr_sq = (dfr_dr * delta_r)**2  # shape (N,)
    del r, delta_r, dfr_dr  # Free memory
    gc.collect()  # Force garbage collection

    # Save the contracted positions
    positions = fr[:, None] * np.column_stack([cos_ra * cos_dec, sin_ra * cos_dec, sin_dec])  # shape (N, 3)
    np.save(file_path_contracted_positions, positions)
    print(f"... saved contracted positions to {file_path_contracted_positions} (shape: {positions.shape})")
    del positions  # Free memory
    gc.collect()  # Force garbage collection

    # Save the contracted position uncertainties
    fr_sq = fr**2  # shape (N,)
    delta_omega_sq = (cos_dec * delta_ra)**2 + delta_dec**2  # shape (N,)

    # RMS 3D position uncertainty
    delta_positions = np.sqrt(delta_fr_sq + fr_sq * delta_omega_sq) # shape (N,)
    np.save(file_path_contracted_position_uncertainties, delta_positions)
    print(f"... saved contracted position uncertainties to {file_path_contracted_position_uncertainties} (shape: {delta_positions.shape})")
    del delta_omega_sq, delta_positions  # Free memory
    gc.collect()  # Force garbage collection

    # Save the contracted velocities
    mu_ra_cos_dec = cos_dec * mu_ra  # shape (N,) in mas/yr
    e_ra = np.column_stack([-sin_ra, cos_ra, np.zeros_like(cos_ra)])  # Tangential basis vector in RA direction
    e_dec = np.column_stack([-cos_ra * sin_dec, -sin_ra * sin_dec, cos_dec])  # Tangential basis vector in Dec direction
    conversion_factor = 149597870.7 / (1000 * 365.25 * 24 * 3600)  # Conversion factor from (pc * mas/yr) to km/s
    velocities = conversion_factor * fr[:, None] * (mu_ra_cos_dec[:, None] * e_ra + mu_dec[:, None] * e_dec)  # shape (N, 3)
    np.save(file_path_contracted_velocities, velocities)
    print(f"... saved contracted velocities to {file_path_contracted_velocities} (shape: {velocities.shape})")

    if WITH_RADIAL_VELOCITIES:
        e_r = np.column_stack([
            cos_dec * cos_ra,
            cos_dec * sin_ra,
            sin_dec
        ])  # Basis vector in radial direction
        radial_velocities = vr[:, None] * e_r
        np.save(file_path_radial_velocities_3D, radial_velocities)
        print(f"... saved radial velocities as 3D vectors to {file_path_radial_velocities_3D} (shape: {radial_velocities.shape})")
        
        np.save(file_path_radial_velocity_uncertainties_3D, delta_vr)  # Save radial velocity uncertainties for later use in data space construction
        print(f"... saved radial velocity uncertainties to {file_path_radial_velocity_uncertainties_3D} (shape: {delta_vr.shape})")
        del e_r, radial_velocities, delta_vr  # Free memory
        gc.collect()  # Force garbage collection
    
    # Save the contracted velocity uncertainties
    mu_magnitude_sq = mu_ra_cos_dec**2 + mu_dec**2  # shape (N,) in (mas/yr)^2
    delta_pm_sq = (
        (mu_ra_cos_dec**2 + (mu_dec * sin_dec)**2) * delta_ra**2 +  # Right ascension component
        ((mu_ra * sin_dec)**2 + mu_dec**2) * delta_dec**2 +         # Declination component
        (cos_dec * delta_mu_ra)**2 +                                # Proper motion in the right ascension component
        (delta_mu_dec)**2                                           # Proper motion in the declination component
    )  # shape (N,) in (mas/yr)^2

    # RMS 3D velocity uncertainty
    delta_velocities = conversion_factor * np.sqrt(delta_fr_sq * mu_magnitude_sq + fr_sq * delta_pm_sq) # shape (N,)
    np.save(file_path_contracted_velocity_uncertainties, delta_velocities)
    print(f"... saved velocity uncertainties to {file_path_contracted_velocity_uncertainties} (shape: {delta_velocities.shape}).\n")
    del cos_ra, cos_dec, sin_ra, sin_dec, mu_ra, mu_dec, delta_ra, delta_dec, delta_mu_ra, delta_mu_dec, fr, delta_fr_sq, fr_sq, mu_ra_cos_dec, mu_magnitude_sq, delta_pm_sq, velocities, delta_velocities, e_ra, e_dec  # Free memory
    gc.collect()  # Force garbage collection

def construct_data_space(overwrite=False):
    """
    Calculate the data space for the subsample.
    """
    # Check if data space already exists
    file_data_space = os.path.join(INTERMEDIATE_FILES_PATH, "data_space.npy")
    if os.path.exists(file_data_space) and not overwrite:
        print(f"Data space already exists at:\n\t{file_data_space} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating data space for subsample...")

    # Load the contracted positions and velocities
    print("... loading required arrays")
    positions = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "contracted_positions.npy"))  # (N, 3)
    velocities = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "contracted_velocities.npy"))  # (N, 3)
    delta_positions = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "contracted_position_uncertainties.npy"))  # (N,)
    delta_velocities = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "contracted_velocity_uncertainties.npy"))  # (N,)
    if WITH_RADIAL_VELOCITIES:
        print('... loading radial velocities and uncertainties for subsample')
        vr = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "radial_velocities_3D.npy"))  # shape (N,) in km/s
        delta_vr = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "radial_velocity_uncertainties_3D.npy"))  # shape (N,) in km/s

    # Calculate scaling factor for positions
    norm_pos = np.median(delta_positions)  # Calculate scaling factor
    print(f"... scaling factor for positions: {norm_pos:.8f}")
    positions /= norm_pos  # Scale positions
    del delta_positions  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate scaling factor for velocities
    norm_vel = np.median(delta_velocities) / RELATIVE_PROPER_MOTION_RESCALE_FACTOR  # Calculate scaling factor
    print(f"... scaling factor for velocities: {norm_vel:.8f}")
    velocities /= norm_vel  # Scale velocities
    del delta_velocities  # Free memory
    gc.collect()  # Force garbage collection

    if WITH_RADIAL_VELOCITIES:
        # Calculate scaling factor for radial velocities
        norm_vr = np.median(delta_vr) / RELATIVE_RADIAL_VELOCITY_RESCALE_FACTOR  # Calculate scaling factor
        print(f"... scaling factor for radial velocities: {norm_vr:.8f}")
        vr /= norm_vr  # Scale radial velocities
        velocities += vr  # Add radial velocity component to the velocity data space
        del vr, delta_vr  # Free memory
        gc.collect()  # Force garbage collection

    # Construct data space for clustering
    print("... constructing data space")
    data_space = np.concatenate([positions, velocities], axis=1)  # shape (N, 6)
    del positions, velocities  # Free memory
    gc.collect()  # Force garbage collection

    # Save data space
    print(f"... saving data space to {file_data_space} (shape: {data_space.shape}).\n")
    np.save(file_data_space, data_space)
    del data_space  # Free memory
    gc.collect()  # Force garbage collection


# === Apply AstroLink to subsample and plot of cluster properties ===
def apply_astrolink_to_data(overwrite=False):
    """
    Run AstroLink clustering on the subsample.
    """
    # Check if AstroLink clustering output already exists
    file_astrolink_object = os.path.join(OUTPUT_PATH, "astrolink_object.npz")
    if os.path.exists(file_astrolink_object) and not overwrite:
        print(f"AstroLink clustering output already exists at:\n\t{file_astrolink_object} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Running AstroLink clustering on the subsample...")

    # Load the required arrays
    print("... loading required arrays for AstroLink clustering")
    input_data = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "data_space.npy"))  # (N, 6)
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N,)
    nsub = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nsub.npy"))[subsample_mask]  # (N,)
    nmw = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "total_selection_function_nmw.npy"))[subsample_mask]  # (N,)
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate inverse total selection function and use as weights
    print("... calculating inverse total selection function to use as weights")
    if STOCHASTIC_RUN:
        # Batched sampling from the posterior distribution of S_total^{-1} ~ 1 + BetaPrime(n_mw - n_sub + 1, n_sub + 1)
        n_rows = nsub.shape[0]
        chunk_n_rows = min(int(WORKING_MEMORY // (8 * 5)), n_rows)  # Number of rows to process in each batch
        weights = np.empty(n_rows)
        for start in range(0, n_rows, chunk_n_rows):
            end = min(start + chunk_n_rows, n_rows)

            # Get batch
            nmw_batch = nmw[start:end]
            nsub_batch = nsub[start:end]

            # Get Gamma shape parameters
            a = nsub_batch + 1
            b = nmw_batch - nsub_batch - 1

            # Sample from the Beta Prime distribution using its Gamma representation
            g1 = np.random.gamma(shape=b, scale=1.0)
            g2 = np.random.gamma(shape=a, scale=1.0)
            weights[start:end] = 1 + (a * g1) / (b * g2)
    else:
        weights = (nmw + 2) / (nsub + 2)  # Mode of the posterior distribution of S_total^{-1} ~ 1 + BetaPrime(n_mw - n_sub + 1, n_sub + 1)
    del nsub, nmw  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate the intrinsic dimensionality of the data based off available astrometric solution components
    d_intrinsic = 5 # Positions and proper motions are always included
    if WITH_RADIAL_VELOCITIES: d_intrinsic += 1

    # Initialize AstroLink
    print("... initializing AstroLink object")
    clusterer = AstroLink(
        P=input_data,
        d_intrinsic=d_intrinsic,
        weights=weights,
        k_den=KNN_FOR_ASTROLINK,
        adaptive=0,
        S=OPTIMAL_SIGMA_THRESHOLD,
        workers=MAX_PARALLEL_WORKERS,
        verbose=0,
        working_memory=WORKING_MEMORY
    )
    del input_data, weights  # Free memory
    gc.collect()  # Force garbage collection

    # The following is a reworked version of the astrolink.run() method
    # It is more memory efficient and has print statements that better align with the rest of the script
    print(f"... [AstroLink] Started             | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    begin = time.perf_counter()

    # Transform the data (this doesn't do anything when adaptive=0, but is required to create the P_transform attribute)
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

def plot_astrolink_prominence_model_fit(overwrite=False):
    """
    Plot the prominence model fit from AstroLink.
    """
    # Check if plots already exist
    file_prominence_model_fit_path = os.path.join(OUTPUT_PATH, "AstroLink_prominence_model_fit.png")
    if os.path.exists(file_prominence_model_fit_path) and not overwrite:
        print(f"Prominence model fit plot already exists at:\n\t{file_prominence_model_fit_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting AstroLink prominence model fit...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(OUTPUT_PATH, "astrolink_object.npz"))
    
    # Plot the prominence model fit
    fig, ax = plt.subplots(figsize=(6, 5))

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
    num_clusters = np.empty_like(SIGMA_THRESHOLDS_FOR_COMPARISONS, dtype=np.int64)
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
    del num_clusters  # Free memory
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
    del clusterer # Free memory
    gc.collect()  # Force garbage collection
    print(f"... saved prominence model fit plot to {file_prominence_model_fit_path}.\n")

def plot_astrolink_cluster_labels_on_sky(overwrite=False):
    """
    Plot the clustering output from AstroLink.
    """
    # Check if plots already exist
    file_clusters_on_sky_path = os.path.join(OUTPUT_PATH, "AstroLink_clusters_on_sky.png")
    if os.path.exists(file_clusters_on_sky_path) and not overwrite:
        print(f"Clusters on sky plot already exists at:\n\t{file_clusters_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting AstroLink clusters on the sky...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(OUTPUT_PATH, "astrolink_object.npz"))

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
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

def plot_astrolink_cluster_proper_motions_on_sky(overwrite=False):
    """
    Plot the proper motions of the clusters on the sky.
    """
    # Check if plots already exist
    file_proper_motions_on_sky_path = os.path.join(OUTPUT_PATH, "AstroLink_cluster_proper_motions_on_sky.png")
    if os.path.exists(file_proper_motions_on_sky_path) and not overwrite:
        print(f"Proper motions on sky plot already exists at:\n\t{file_proper_motions_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting AstroLink clusters' proper motions on the sky...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(OUTPUT_PATH, "astrolink_object.npz"))

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    equatorial_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_equatorial_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    proper_motions = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_proper_motions.npy"))[subsample_mask]  # (N, 2) in mas/yr
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
    mu_magnitude = np.clip(np.ceil(8 * mu_magnitude) / 8, 0, 1)  # Bin into 8 discrete magnitudes for better colour contrast
    mu_magnitude = mu_magnitude**0.8  # Adjust brightness scaling for better visibility
    mu_angle = np.arctan2(mu_b, mu_l_cosb)  # Proper motion angle in radians
    mu_angle = (mu_angle + np.pi) / (2 * np.pi)  # Shift to [0, 2*pi] range
    mu_angle = (np.floor(16 * mu_angle) + 0.5) / 16  # Bin into 16 discrete angles for better colour contrast
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
    N = 2**9
    radius = 1
    y, x = np.ogrid[-radius:radius:N*1j, -radius:radius:N*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    del y, x  # Free memory
    gc.collect()  # Force garbage collection

    # --- Apply same transformations as for the stars ---

    # Angle → hue
    hue = (theta + np.pi) / (2 * np.pi)             # [0, 1]
    hue = (np.floor(16 * hue) + 0.5) / 16           # 16 discrete angle bins (midpoints)

    # Radius → saturation
    saturation = np.clip(r, 0, 1)                   # [0, 1]
    saturation = np.clip(np.ceil(8 * saturation) / 8, 0, 1)  # 8 discrete magnitude bins (outer edges)
    saturation = saturation ** 0.8                  # Brightness scaling

    # Value (brightness)
    value = np.ones_like(hue)

    # Stack and convert to RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)

    del hue, saturation, value, hsv
    gc.collect()  # Force garbage collection

    # Add alpha channel
    alpha = np.ones((N, N, 1))
    rgba = np.concatenate([rgb, alpha], axis=-1)
    del rgb, alpha
    gc.collect()

    # Mask outside the circle
    mask = r > 1
    rgba[mask] = 0  # Clear outside the circle

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

def plot_astrolink_cluster_radial_velocities_on_sky(overwrite=False):
    """
    Plot the clustering output from AstroLink.
    """
    if not WITH_RADIAL_VELOCITIES: return
    # Check if plots already exist
    file_radial_velocities_on_sky_path = os.path.join(OUTPUT_PATH, "AstroLink_cluster_radial_velocities_on_sky.png")
    if os.path.exists(file_radial_velocities_on_sky_path) and not overwrite:
        print(f"Radial velocities on sky plot already exists at:\n\t{file_radial_velocities_on_sky_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting AstroLink clusters' radial velocities on the sky...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(OUTPUT_PATH, "astrolink_object.npz"))

    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (n,)
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    radial_velocities = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_radial_velocities.npy"))[subsample_mask]  # (N,) in km/s
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

    # Calculate radial velocity colours for plotting
    print("... calculating radial velocity for plotting")
    vr_clipped = np.clip(radial_velocities, -50, 50) / 50  # Clip to avoid extreme values
    vr_clipped = np.clip((np.ceil(8 * vr_clipped) - 0.5) / 8, -1, 1)  # Bin into 16 discrete magnitudes for better colour contrast
    vr_clipped += 1
    vr_clipped /= 2  # in [0, 1]
    cmap = plt.colormaps["coolwarm"]
    colours = cmap(vr_clipped)
    del radial_velocities, vr_clipped  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot clusters on the sky
    print("... creating Mollweide projection plot for clusters' radial velocities on the sky")
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    for i, clst in enumerate(clusterer.clusters[1:]):
        clusterMembers = clusterer.ordering[clst[0]:clst[1]]
        ax.scatter(
            *galactic_coordinates[clusterMembers].T,
            facecolor=colours[clusterMembers], edgecolor='k',
            s=0.75, lw=0.075
        )  # Plot each cluster with colours according to their proper motions
    del clusterer, galactic_coordinates, colours  # Free memory
    gc.collect()  # Force garbage collection

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_radial_velocities_on_sky_path, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters' radial velocities on sky plot to {file_radial_velocities_on_sky_path}.\n")

# === Define reusable methods for comparing to and plotting existing catalogues ===
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

def plot_catalogue_structure_on_sky(
        members_cluster_ids_subsample,
        members_cluster_membership_subsample,
        file_path_clusters_on_sky
    ):
    """
    Plot the structure of a catalogue on the sky.
    """
    # Load the required arrays
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N,)
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]  # (N, 2) in degrees
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Convert (l, b) in degrees to radians for Mollweide projection
    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)

    # Mollweide expects longitudes in the range [-pi, pi] and latitudes in the range [-pi/2, pi/2]
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1 # Invert x-axis for on-sky astro plot

    # Simplify the catalogue
    print("... simplifying Hunt & Reffert (2024) catalogue for plotting")
    plottable_bool = np.any(members_cluster_membership_subsample > 0.5, axis=1)
    no_cluster_id = members_cluster_ids_subsample.max()  # Define "no cluster" ID
    members_cluster_ids_subsample = members_cluster_ids_subsample[plottable_bool]
    galactic_coordinates = galactic_coordinates[plottable_bool]
    print(f"... {plottable_bool.sum()} plottable stars")
    del members_cluster_membership_subsample, plottable_bool  # Free memory
    gc.collect()  # Force garbage collection

    # Create a Mollweide projection plot and plot clusters on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    print("... plotting clusters on the sky")
    for i in range(no_cluster_id):
        clusterMembers = np.any(members_cluster_ids_subsample == i, axis=1)

        # Filter out excess members for better plotting
        n_cluster = clusterMembers.sum()
        if n_cluster > 5000:
            clusterMembers_indices = np.where(clusterMembers)[0]
            removed_indices = np.random.choice(clusterMembers_indices, size=n_cluster-5000, replace=False)
            clusterMembers[removed_indices] = False

        # Plot the cluster members
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
    plt.savefig(file_path_clusters_on_sky, dpi=500)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters on sky plot to {file_path_clusters_on_sky}.\n")

def compare_to_catalogue_helper(
        members_cluster_ids_subsample,
        members_cluster_membership_subsample,
        cluster_sizes_in_subsample,
        comparison_metric,
        file_path_catalogue_rpj_allS,
        file_path_catalogue_idx_optimalS,
        file_path_astrolink_rpj_optimalS,
        file_path_astrolink_idx_optimalS,
        file_path_catalogue_rpj_optimalS
    ):
    # Max cluster ID in the catalogue (this + 1 is the "no cluster" ID")
    max_catalogue_cluster_ID = members_cluster_ids_subsample.max() - 1 # Assumes that at least one star in the subsample isn't a member of any cluster (i.e. has "no cluster" ID)

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(OUTPUT_PATH, "astrolink_object.npz"))

    # Put large arrays into shared memory
    print("... putting large arrays into shared memory")
    shm_ids, shape_ids, dtype_ids = arr_to_shared_memory(members_cluster_ids_subsample)
    shm_membership, shape_membership, dtype_membership = arr_to_shared_memory(members_cluster_membership_subsample)
    shm_ordering, shape_ordering, dtype_ordering = arr_to_shared_memory(clusterer.ordering)

    # Use limited number of workers because this process is memory intensive
    max_workers = min(8, MAX_PARALLEL_WORKERS)

    # Initialise array to store crossmatching statistics for AstroLink leaf clusters relative to the catalogue clusters across all significance thresholds
    catalogue_rpj_allS = np.zeros((SIGMA_THRESHOLDS_FOR_COMPARISONS.size, cluster_sizes_in_subsample.size, 5), dtype=np.float32)  # (N_sigmas, N_clusters, 5) to store RPJ values
    
    # Loop over significance values
    for k, significance in enumerate(SIGMA_THRESHOLDS_FOR_COMPARISONS):
        print(f"... calculating cluster-match statistics at significance level S={significance:.1f}   ", end = '\r')
        # Extract clusters at the current significance level
        clusterer.S = significance
        clusterer.extract_clusters()

        # Check if at the optimal significance threshold
        isOptimalSignificance = significance == OPTIMAL_SIGMA_THRESHOLD

        # Initialise arrays to store best matches at the optimal significance threshold
        if isOptimalSignificance:
            catalogue_idx_optimalS = -np.ones(clusterer.clusters.shape[0], dtype=np.int64)  # (N_AstroLink_clusters,) to store best matching catalogue cluster indices per AstroLink cluster
            astrolink_rpj_optimalS = np.zeros((clusterer.clusters.shape[0], 5), dtype=np.float32)  # (N_AstroLink_clusters, 5) to store best matching RPJ values per AstroLink cluster
            astrolink_idx_optimalS = -np.ones((cluster_sizes_in_subsample.size,), dtype=np.int64)  # (N_catalogue_clusters,) to store best matching AstroLink cluster indices per catalogue cluster
            catalogue_rpj_optimalS = np.zeros((cluster_sizes_in_subsample.size, 5), dtype=np.float32)  # (N_catalogue_clusters, 5) to store best matching RPJ values per catalogue cluster

        # Loop over AstroLink clusters in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for cluster_index, (start, end) in enumerate(clusterer.clusters):
                # Check if this AstroLink cluster is a leaf cluster
                isLeafCluster = cluster_index == clusterer.ids.size - 1 or not clusterer.ids[cluster_index + 1].startswith(clusterer.ids[cluster_index] + '-')
                
                # Compare leaf clusters only or all clusters if at optimal signficance (but never the trivial background cluster)
                if cluster_index > 0 and (isLeafCluster or isOptimalSignificance): 
                    futures.append(
                        executor.submit(process_astrolink_cluster,
                                        cluster_index, isLeafCluster, start, end,
                                        cluster_sizes_in_subsample,
                                        max_catalogue_cluster_ID,
                                        shm_ids.name, shm_membership.name, shm_ordering.name,
                                        shape_ids, shape_membership, shape_ordering,
                                        dtype_ids, dtype_membership, dtype_ordering)
                    )

            # Process comparison results as they finish
            for f in as_completed(futures):
                result = f.result()
                if result is None:
                    continue
                
                # Unpack result
                cluster_index, isLeafCluster, unique_ids, single_RPJ = result

                # Record RPJ values for all clusters at the optimal significance (except background cluster)
                if isOptimalSignificance:
                    # Crossmatching relative to this AstroLink cluster, i.e. what is the best matching catalogue cluster to the AstroLink clusters?
                    best_match_boolean = np.argmax(single_RPJ[:, comparison_metric]) # Best catalogue match to this AstroLink cluster
                    best_match_index   = unique_ids[best_match_boolean]
                    catalogue_idx_optimalS[cluster_index] = best_match_index
                    astrolink_rpj_optimalS[cluster_index] = single_RPJ[best_match_boolean]

                    # Crossmatching relative to the catalogue clusters, i.e. what is the best matching AstroLink cluster to the catalogue clusters?
                    better_matches_boolean = single_RPJ[:, comparison_metric] > catalogue_rpj_optimalS[unique_ids, comparison_metric] # Comparison under the chosen metric
                    better_matches_indices = unique_ids[better_matches_boolean]
                    astrolink_idx_optimalS[better_matches_indices] = cluster_index
                    catalogue_rpj_optimalS[better_matches_indices] = single_RPJ[better_matches_boolean]

                # Record RPJ values for leaf clusters only (relative to the catalogue clusters only)
                if isLeafCluster:
                    better_matches_boolean = single_RPJ[:, comparison_metric] > catalogue_rpj_allS[k, unique_ids, comparison_metric] # Comparison under the chosen metric
                    better_matches_indices = unique_ids[better_matches_boolean]
                    catalogue_rpj_allS[k, better_matches_indices] = single_RPJ[better_matches_boolean]
    
    # Clean up shared memory
    print("... cleaning up shared memory                                                                                                  ")
    shm_ids.close(); shm_ids.unlink()
    shm_membership.close(); shm_membership.unlink()
    shm_ordering.close(); shm_ordering.unlink()
    del members_cluster_ids_subsample, members_cluster_membership_subsample, cluster_sizes_in_subsample, clusterer  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_catalogue_rpj_allS, catalogue_rpj_allS)
    np.save(file_path_catalogue_idx_optimalS, catalogue_idx_optimalS)
    np.save(file_path_astrolink_rpj_optimalS, astrolink_rpj_optimalS)
    np.save(file_path_astrolink_idx_optimalS, astrolink_idx_optimalS)
    np.save(file_path_catalogue_rpj_optimalS, catalogue_rpj_optimalS)
    del catalogue_rpj_allS, catalogue_idx_optimalS, astrolink_rpj_optimalS, astrolink_idx_optimalS, catalogue_rpj_optimalS  # Free memory
    gc.collect()  # Force garbage collection

def arr_to_shared_memory(arr):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype.type, buffer=shm.buf)
    np.copyto(shm_arr, arr)
    return shm, arr.shape, arr.dtype.type

def process_astrolink_cluster(
        cluster_index, isLeafCluster, start, end,
        cluster_sizes_in_subsample,
        max_catalogue_cluster_ID,
        shm_name_ids, shm_name_membership, shm_name_ordering,
        shape_ids, shape_membership, shape_ordering,
        dtype_ids, dtype_membership, dtype_ordering
    ):
    """
    Worker function to process one AstroLink cluster.
    Reattaches shared-memory arrays, extracts cluster members, and computes RPJ stats.
    Returns (updates to whichAstroLinkClusters, RPJ).
    """
    # Reattach shared-memory arrays
    shm_ids = shared_memory.SharedMemory(name=shm_name_ids)
    shm_membership = shared_memory.SharedMemory(name=shm_name_membership)
    shm_ordering = shared_memory.SharedMemory(name=shm_name_ordering)

    members_cluster_ids_subsample = np.ndarray(shape_ids, dtype=dtype_ids, buffer=shm_ids.buf)
    members_cluster_membership_subsample = np.ndarray(shape_membership, dtype=dtype_membership, buffer=shm_membership.buf)
    ordering = np.ndarray(shape_ordering, dtype=dtype_ordering, buffer=shm_ordering.buf)

    # Get cluster members in the AstroLink output
    astrolink_cluster_members = ordering[start:end]

    # Get cluster IDs for those members
    xmatched_cluster_IDs = members_cluster_ids_subsample[astrolink_cluster_members]

    # Flatten IDs
    ids_flat = xmatched_cluster_IDs.ravel()

    # Handle "no cluster" ID = max_catalogue_cluster_ID + 1
    mask_valid = ids_flat <= max_catalogue_cluster_ID
    ids_valid = ids_flat[mask_valid]

    if ids_valid.size == 0:
        return None  # No valid clusters to compare

    # If there are clusters to compare to, get valid members
    xmatched_cluster_membership = members_cluster_membership_subsample[astrolink_cluster_members]
    membership_flat = xmatched_cluster_membership.ravel()
    membership_valid = membership_flat[mask_valid]

    # Get unique IDs and their associated intersections
    unique_ids, inv = np.unique(ids_valid, return_inverse=True)
    intersection_sizes = np.bincount(inv, weights=membership_valid)

    # Call numba-jitted function
    RPJ_cluster = calculate_rpj_for_astrolink_cluster_matches(
        unique_ids,
        cluster_sizes_in_subsample,
        intersection_sizes,
        xmatched_cluster_IDs,
        max_catalogue_cluster_ID
    )

    return (cluster_index, isLeafCluster, unique_ids, RPJ_cluster)

@njit
def calculate_rpj_for_astrolink_cluster_matches(
        unique_ids,
        cluster_sizes_in_subsample,
        intersection_sizes,
        xmatched_cluster_IDs,
        max_catalogue_cluster_ID
    ):
    n = unique_ids.size
    RPJ = np.empty((n, 5), dtype=np.float64)

    # Sizes
    Ai_true_size = xmatched_cluster_IDs.shape[0]

    not_in_catalogue = np.sum(
        xmatched_cluster_IDs[:, 0] == max_catalogue_cluster_ID + 1
    )
    Ai_overlap_size = Ai_true_size - not_in_catalogue

    # Vector retrieval (safe)
    Cj_apparent = cluster_sizes_in_subsample[unique_ids]

    # Precompute denominators
    denom_rec = Cj_apparent.astype(np.float64)
    denom_pur_SA = float(Ai_true_size)
    denom_pur_union = float(Ai_overlap_size)

    denom_jac_SA = denom_pur_SA + denom_rec - intersection_sizes
    denom_jac_union = denom_pur_union + denom_rec - intersection_sizes

    # Now explicit small loop just for safe division
    for i in range(n):
        inter = intersection_sizes[i]

        # Recovery
        RPJ[i, 0] = inter / denom_rec[i] if denom_rec[i] > 0 else np.nan

        # Purity (S_C = S_A)
        RPJ[i, 1] = inter / denom_pur_SA if denom_pur_SA > 0 else np.nan

        # Purity (S_C = union)
        RPJ[i, 2] = inter / denom_pur_union if denom_pur_union > 0 else np.nan

        # Jaccard (S_C = S_A)
        RPJ[i, 3] = inter / denom_jac_SA[i] if denom_jac_SA[i] > 0 else np.nan

        # Jaccard (S_C = union)
        RPJ[i, 4] = inter / denom_jac_union[i] if denom_jac_union[i] > 0 else np.nan

    return RPJ

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
    # Check if Hunt2024 folder exists in INTERMEDIATE_FILES_PATH
    hunt2024_path = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024")
    os.makedirs(hunt2024_path, exist_ok=True)

    # Check if files already exist
    file_path_clusters_names = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_clusters_names.npy")
    file_path_clusters_types = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_clusters_types.npy")
    file_path_clusters_snr = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_clusters_snr.npy")
    file_path_H24_members_cluster_ids_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_members_cluster_ids_subsample.npy")
    file_path_H24_members_cluster_membership_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_members_cluster_membership_subsample.npy")
    file_path_H24_cluster_sizes_in_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_cluster_sizes_in_subsample.npy")
    file_path_H24_cluster_coverage = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_cluster_coverage.npy")

    # Skip if all files exist and overwrite is False
    all_exist = (os.path.exists(file_path_clusters_names) and
                 os.path.exists(file_path_clusters_types) and
                 os.path.exists(file_path_clusters_snr) and
                 os.path.exists(file_path_H24_members_cluster_ids_subsample) and
                 os.path.exists(file_path_H24_members_cluster_membership_subsample) and
                 os.path.exists(file_path_H24_cluster_sizes_in_subsample) and
                 os.path.exists(file_path_H24_cluster_coverage))
    if all_exist and not overwrite:
        print("Hunt & Reffert (2024) reduced data already exists at:")
        print(f"\t{file_path_clusters_names} ,")
        print(f"\t{file_path_clusters_types} ,")
        print(f"\t{file_path_clusters_snr} ,")
        print(f"\t{file_path_H24_members_cluster_ids_subsample} ,")
        print(f"\t{file_path_H24_members_cluster_membership_subsample} ,")
        print(f"\t{file_path_H24_cluster_sizes_in_subsample} , and")
        print(f"\t{file_path_H24_cluster_coverage} .")
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

    # Convert probabilistic membership to a binary one
    H24_members_cluster_membership = H24_members_probs >= 0.5

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Load subsample mask
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N,)

    # Save the membership mask
    print("... making membership mask")
    indices = np.searchsorted(gdr3_source_ids, H24_members_source_ids) # Assumes gdr3_source_ids is sorted
    H24_members_mask = np.zeros_like(gdr3_source_ids, dtype=np.bool_)  # Create a mask of the same shape as gdr3_source_ids
    H24_members_mask[indices] = True  # Set the indices of the members to True
    del gdr3_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Get Hunt+2024 cluster IDs for this subsample (multiple columns since stars can be in multiple Hunt+2024 clusters)
    print("... getting Hunt+2024 cluster IDs and memberships for the subsample in this work")
    max_appearances = np.unique(H24_members_source_ids, return_counts=True)[1].max()
    max_H24_cluster_ID = H24_members_cluster_ids.max()
    H24_members_cluster_ids_gdr3 = np.full((H24_members_mask.size, max_appearances), max_H24_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_H24_cluster_ID + 1, representing no cluster
    H24_members_cluster_membership_gdr3 = np.zeros((H24_members_mask.size, max_appearances), dtype=np.bool_)  # (N,) Initialize with 0, representing zero membership probability
    H24_members_mask_where = np.where(H24_members_mask)[0]  # Use indices of members in the full catalogue from now on to do efficient slicing/indexing
    del H24_members_mask  # Free memory
    gc.collect()  # Force garbage collection
    
    # Assign cluster IDs / memberships to each of the members
    unique_H24_source_ids, indices, counts = np.unique(H24_members_source_ids, return_index=True, return_counts=True)
    for num_count in range(1, max_appearances + 1):
        # Get the relative position of the members in the full catalogue for this count
        num_count_bool = counts == num_count

        # Get the cluster IDs and memberships for the members with this count
        if num_count == 1:
            cluster_ids = H24_members_cluster_ids[indices[num_count_bool]][:, None]
            membership = H24_members_cluster_membership[indices[num_count_bool]][:, None]
        else: # There are not too many of these so what follows is efficient enough
            cluster_ids = np.zeros((num_count_bool.sum(), num_count), dtype=np.int64)
            membership = np.zeros((num_count_bool.sum(), num_count), dtype=np.bool_)
            for i, sid in enumerate(unique_H24_source_ids[num_count_bool]):
                # Get the indices of the members with this source ID
                source_id_match = np.where(H24_members_source_ids == sid)[0]

                # Get the cluster IDs and memberships for these members
                cluster_ids[i] = H24_members_cluster_ids[source_id_match]
                membership[i] = H24_members_cluster_membership[source_id_match]

        # Assign the cluster IDs and memberships to the members
        H24_members_cluster_ids_gdr3[H24_members_mask_where[num_count_bool], :num_count] = cluster_ids
        H24_members_cluster_membership_gdr3[H24_members_mask_where[num_count_bool], :num_count] = membership

    H24_members_cluster_ids_subsample = H24_members_cluster_ids_gdr3[subsample_mask]
    H24_members_cluster_membership_subsample = H24_members_cluster_membership_gdr3[subsample_mask]
    del subsample_mask, H24_members_source_ids, H24_members_cluster_ids_gdr3, H24_members_cluster_membership_gdr3, H24_members_mask_where, unique_H24_source_ids, indices, counts  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total size for each Hunt+2024 cluster
    print("... pre-computing the total size for each Hunt+2024 cluster")
    H24_cluster_sizes_full = np.bincount(
        H24_members_cluster_ids,
        weights=H24_members_cluster_membership,
        minlength=max_H24_cluster_ID + 1
    )  # (N_clusters,)
    del H24_members_cluster_ids, H24_members_cluster_membership  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the size for each Hunt+2024 cluster in the overlap with the subsample
    print("... pre-computing the size for each Hunt+2024 cluster in the subsample of this work")
    H24_cluster_sizes_in_subsample = np.bincount(
        H24_members_cluster_ids_subsample.ravel(),
        weights=H24_members_cluster_membership_subsample.ravel(),
        minlength=max_H24_cluster_ID + 1
    )[:max_H24_cluster_ID + 1]  # (N_clusters,)

    # Calculate the coverage of each Hunt+2024 cluster by the subsample of this work
    H24_cluster_coverage = np.full_like(H24_cluster_sizes_full, 0.0, dtype=np.float64)
    valid = H24_cluster_sizes_full > 0
    H24_cluster_coverage[valid] = (
        H24_cluster_sizes_in_subsample[valid] /
        H24_cluster_sizes_full[valid]
    )
    del H24_cluster_sizes_full  # Free memory
    gc.collect()  # Force garbage collection

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_H24_members_cluster_ids_subsample, H24_members_cluster_ids_subsample)
    np.save(file_path_H24_members_cluster_membership_subsample, H24_members_cluster_membership_subsample)
    np.save(file_path_H24_cluster_sizes_in_subsample, H24_cluster_sizes_in_subsample)
    np.save(file_path_H24_cluster_coverage, H24_cluster_coverage)
    del H24_members_cluster_ids_subsample, H24_members_cluster_membership_subsample, H24_cluster_sizes_in_subsample, H24_cluster_coverage  # Free memory
    gc.collect()  # Force garbage collection

def plot_Hunt2024_clusters_on_sky(overwrite=False):
    """
    Plot the Hunt & Reffert (2024) clusters on the sky.
    """
    # Check if plot already exists
    file_path_clusters_on_sky = os.path.join(OUTPUT_PATH, "Hunt2024_clusters_on_sky.png")
    if os.path.exists(file_path_clusters_on_sky) and not overwrite:
        print(f"Hunt & Reffert (2024) clusters plot already exists at:\n\t{file_path_clusters_on_sky} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Hunt & Reffert (2024) clusters...")

    # Load the Hunt & Reffert (2024) clustering output
    print("... loading Hunt & Reffert (2024) catalogue")
    H24_members_cluster_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_members_cluster_ids_subsample.npy"))  # (N_subsample, max_appearances)
    H24_members_cluster_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_members_cluster_membership_subsample.npy"))  # (N_subsample, max_appearances)

    # Plot the structure on the sky
    plot_catalogue_structure_on_sky(
        H24_members_cluster_ids_subsample,
        H24_members_cluster_membership_subsample,
        file_path_clusters_on_sky
    )

def compare_to_Hunt2024(overwrite=False):
    """
    Compare the clustering output to the Hunt & Reffert (2024).
    """
    # Check if comparison results already exist
    file_path_catalogue_rpj_allS = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_catalogue_rpj_allS.npy")
    file_path_catalogue_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_catalogue_idx_optimalS.npy")
    file_path_astrolink_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_astrolink_rpj_optimalS.npy")
    file_path_astrolink_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_astrolink_idx_optimalS.npy")
    file_path_catalogue_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_catalogue_rpj_optimalS.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_catalogue_rpj_allS) and
                 os.path.exists(file_path_catalogue_idx_optimalS) and
                 os.path.exists(file_path_astrolink_rpj_optimalS) and
                 os.path.exists(file_path_astrolink_idx_optimalS) and
                 os.path.exists(file_path_catalogue_rpj_optimalS))
    if all_exist and not overwrite:
        print("Hunt & Reffert (2024) comparison results already exist at:")
        print(f"\t{file_path_catalogue_rpj_allS} ,")
        print(f"\t{file_path_catalogue_idx_optimalS} ,")
        print(f"\t{file_path_astrolink_rpj_optimalS} ,")
        print(f"\t{file_path_astrolink_idx_optimalS} , and")
        print(f"\t{file_path_catalogue_rpj_optimalS} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to Hunt & Reffert (2024)...")

    # Load required arrays
    print("... loading required arrays for comparison")
    H24_members_cluster_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_members_cluster_ids_subsample.npy"))  # (N_subsample, max_appearances)
    H24_members_cluster_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_members_cluster_membership_subsample.npy"))  # (N_subsample, max_appearances)
    H24_cluster_sizes_in_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_cluster_sizes_in_subsample.npy"))  # (N_clusters,)

    # Choose comparison metric
    comparison_metric = 3  # 0 for recovery, 1 for purity (S_C=S_A), 2 for purity (S_C=all_catalogue_stars), 3 for Jaccard index (S_C=S_A), 4 for Jaccard index (S_C=all_catalogue_stars)
    
    # Make the comparison
    compare_to_catalogue_helper(
        H24_members_cluster_ids_subsample,
        H24_members_cluster_membership_subsample,
        H24_cluster_sizes_in_subsample,
        comparison_metric,
        file_path_catalogue_rpj_allS,
        file_path_catalogue_idx_optimalS,
        file_path_astrolink_rpj_optimalS,
        file_path_astrolink_idx_optimalS,
        file_path_catalogue_rpj_optimalS
    )

def plot_Hunt2024_crossmatch_per_significance(overwrite=False):
    """
    Plot a summary the comparison between the AstroLink clustering output and the 
    Hunt & Reffert (2024) catalogue as a function of the significance threshold S.
    """
    # Check if plot already exists
    file_path = os.path.join(OUTPUT_PATH, "Hunt2024_crossmatch_per_significance.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Hunt & Reffert (2024) crossmatch per significance plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Hunt & Reffert (2024) crossmatch per significance...")

    # Load comparison results
    print("... loading comparison results")
    RPJ = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_catalogue_rpj_allS.npy"))  # (N_sigmas, N_clusters, 5)
    H24_cluster_coverage = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_cluster_coverage.npy"))

    # Load Hunt & Reffert (2024) cluster types
    print("... loading Hunt & Reffert (2024) cluster types")
    H24_cluster_types = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_clusters_types.npy"), allow_pickle=True)  # (N_clusters,)

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # ========== INDIVIDUAL CLUSTER TYPES ==========
    print("... plotting per-cluster-type statistics")
    cluster_type_and_colour = {
        'o': 'C0',  # Open clusters
        'm': 'C2',  # Moving groups
        'g': 'C1',  # Globular clusters
        'd': 'C4',  # Too distant to classify
        'r': 'C3'   # Rejected
    }

    for cluster_type, type_colour in cluster_type_and_colour.items():
        mask = H24_cluster_types == cluster_type
        if not np.any(mask):
            continue

        # Extract and weight-average curves
        J_full, J_union = RPJ[:, mask, 3], RPJ[:, mask, 4]
        mask_coverage = H24_cluster_coverage[mask]

        # Coverage-weighted average
        mask_coverage_sum = np.sum(mask_coverage)
        Jbar_full  = np.sum(J_full * mask_coverage, axis=1) / mask_coverage_sum
        Jbar_union = np.sum(J_union * mask_coverage, axis=1) / mask_coverage_sum

        # Plot Jaccard index curves
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
                color=type_colour, linestyle='solid', linewidth=1, zorder=2)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
                color=type_colour, linestyle='solid', linewidth=1, zorder=2)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                color=type_colour, alpha=0.3, zorder=2)

        # Compute and plot matched fraction curves
        fraction_matched_union = np.mean(J_union > 0.5, axis=1)
        fraction_matched_full = np.mean(J_full > 0.5, axis=1)

        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_union,
                color=type_colour, linestyle='dashed', linewidth=1, zorder=2)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full,
                color=type_colour, linestyle='dashed', linewidth=1, zorder=2)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full, fraction_matched_union,
                facecolor='none', hatch='//', edgecolor=type_colour, linewidth=1, zorder=2)

    # Plot optimal S value on top
    ax.axvline(OPTIMAL_SIGMA_THRESHOLD, color='grey', linestyle='dotted', linewidth=0.75, zorder=4)
    ax.text(
        OPTIMAL_SIGMA_THRESHOLD, 1.01,
        f"S = {OPTIMAL_SIGMA_THRESHOLD}",
        color='grey', fontsize=10, ha='center', va='bottom', zorder=4
    )

    # ========= Final formatting ==========
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Significance, $S$")
    ax.set_ylabel("Comparison statistic")

    # Dummy handles for the legend
    handles = [
        Rectangle((0,0), 1, 1, facecolor=mcolors.to_rgba('k', alpha=0.3), edgecolor='k', linewidth=1, label='Jaccard index'),
        Rectangle((0,0), 1, 1, facecolor='none', hatch='//', edgecolor='k', linewidth=1, label='Matched fraction'),
        plt.Line2D([], [], color='none', label='Open clusters'),
        plt.Line2D([], [], color='none', label='Moving groups'),
        plt.Line2D([], [], color='none', label='Globular clusters'),
        plt.Line2D([], [], color='none', label='Too distant'),
        plt.Line2D([], [], color='none', label='Rejected'),
    ]

    # Create the legend
    leg = ax.legend(handles=handles, loc='upper right', frameon=False)

    # Define cluster type colors
    cluster_type_colors = {
        'Open clusters': 'C0',
        'Moving groups': 'C2',
        'Globular clusters': 'C1',
        'Too distant': 'C4',
        'Rejected': 'C3'
    }

    # Recolor legend text entries for cluster types
    for text in leg.get_texts():
        label = text.get_text()
        if label in cluster_type_colors:
            text.set_color(cluster_type_colors[label])
            text.set_fontweight('bold')

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
    # Check if UCC folder exists in INTERMEDIATE_FILES_PATH
    ucc_path = os.path.join(INTERMEDIATE_FILES_PATH, "UCC")
    os.makedirs(ucc_path, exist_ok=True)

    # Check if files already exist
    file_path_clusters_names = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_clusters_names.npy")
    file_path_clusters_quality_class = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_clusters_quality_class.npy")
    file_path_UCC_members_cluster_ids_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_members_cluster_ids_subsample.npy")
    file_path_UCC_members_cluster_membership_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_members_cluster_membership_subsample.npy")
    file_path_UCC_cluster_sizes_in_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_cluster_sizes_in_subsample.npy")
    file_path_UCC_cluster_coverage = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_cluster_coverage.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_clusters_names) and
                 os.path.exists(file_path_clusters_quality_class) and
                 os.path.exists(file_path_UCC_members_cluster_ids_subsample) and
                 os.path.exists(file_path_UCC_members_cluster_membership_subsample) and
                 os.path.exists(file_path_UCC_cluster_sizes_in_subsample) and
                 os.path.exists(file_path_UCC_cluster_coverage))
    if all_exist and not overwrite:
        print("Unified Cluster Catalogue reduced data already exists at:")
        print(f"\t{file_path_clusters_names} ,")
        print(f"\t{file_path_clusters_quality_class} ,")
        print(f"\t{file_path_UCC_members_cluster_ids_subsample} ,")
        print(f"\t{file_path_UCC_members_cluster_membership_subsample} ,")
        print(f"\t{file_path_UCC_cluster_sizes_in_subsample} , and")
        print(f"\t{file_path_UCC_cluster_coverage} .")
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

    # Convert probabilistic membership to a binary one
    UCC_members_cluster_membership = UCC_members_probs >= 0.5

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Load required arrays
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N_gdr3,)

    # Save the membership mask
    print("... making membership mask")
    indices = np.searchsorted(gdr3_source_ids, UCC_members_source_ids) # Assumes gdr3_source_ids is sorted
    UCC_members_mask = np.zeros_like(gdr3_source_ids, dtype=np.bool_)  # Create a mask of the same shape as gdr3_source_ids
    UCC_members_mask[indices] = True  # Set the indices of the members to True
    del gdr3_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection
    
    # Get UCC cluster IDs for this subsample (multiple columns since stars can be in multiple UCC clusters)
    print("... getting UCC cluster IDs for the subsample in this work")
    max_appearances = np.unique(UCC_members_source_ids, return_counts=True)[1].max()
    unique_UCC_members_cluster_names, UCC_members_cluster_ids = np.unique(UCC_members_cluster_names, return_inverse=True)
    max_UCC_cluster_ID = UCC_members_cluster_ids.max()
    UCC_members_cluster_ids_gdr3 = np.full((UCC_members_mask.size, max_appearances), max_UCC_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_UCC_cluster_ID + 1, representing no cluster
    UCC_members_cluster_membership_gdr3 = np.zeros((UCC_members_mask.size, max_appearances), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
    UCC_members_mask_where = np.where(UCC_members_mask)[0]  # Use indices of members in the full catalogue from now on to do efficient slicing/indexing
    del UCC_members_mask, UCC_members_cluster_names, unique_UCC_members_cluster_names  # Free memory
    gc.collect()  # Force garbage collection

    # Assign the cluster IDs to each of the members
    unique_UCC_source_ids, indices, counts = np.unique(UCC_members_source_ids, return_index=True, return_counts=True)
    for num_count in range(1, max_appearances + 1):
        # Get the relative position of the members in the full catalogue for this count
        num_count_bool = counts == num_count

        # Get the cluster IDs for the members with this count
        if num_count == 1:
            cluster_ids = UCC_members_cluster_ids[indices[num_count_bool]][:, None]
            membership = UCC_members_cluster_membership[indices[num_count_bool]][:, None]
        else: # There are not too many of these so what follows is efficient enough
            cluster_ids = np.zeros((num_count_bool.sum(), num_count), dtype=np.int64)
            membership = np.zeros((num_count_bool.sum(), num_count), dtype=np.float32)
            for i, sid in enumerate(unique_UCC_source_ids[num_count_bool]):
                # Get the indices of the members with this source ID
                source_id_match = np.where(UCC_members_source_ids == sid)[0]

                # Get the cluster IDs for these members
                cluster_ids[i] = UCC_members_cluster_ids[source_id_match]
                membership[i] = UCC_members_cluster_membership[source_id_match]

        # Assign the cluster IDs to the members
        UCC_members_cluster_ids_gdr3[UCC_members_mask_where[num_count_bool], :num_count] = cluster_ids
        UCC_members_cluster_membership_gdr3[UCC_members_mask_where[num_count_bool], :num_count] = membership

    UCC_members_cluster_ids_subsample = UCC_members_cluster_ids_gdr3[subsample_mask]
    UCC_members_cluster_membership_subsample = UCC_members_cluster_membership_gdr3[subsample_mask]
    del subsample_mask, UCC_members_source_ids, UCC_members_cluster_ids_gdr3, UCC_members_mask_where, unique_UCC_source_ids, indices, counts  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total size for each UCC cluster
    print("... pre-computing the total size for each UCC cluster")
    UCC_cluster_sizes_full = np.bincount(
        UCC_members_cluster_ids,
        weights=UCC_members_cluster_membership,
        minlength=max_UCC_cluster_ID + 1
    )  # (N_clusters,)
    del UCC_members_cluster_ids, UCC_members_cluster_membership  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the size for each UCC cluster in the overlap with the subsample
    print("... pre-computing the size for each UCC cluster in the subsample of this work")
    UCC_cluster_sizes_in_subsample = np.bincount(
        UCC_members_cluster_ids_subsample.ravel(),
        weights=UCC_members_cluster_membership_subsample.ravel(),
        minlength=max_UCC_cluster_ID + 1
    )[:max_UCC_cluster_ID + 1]  # (N_clusters,)

    # Calculate the coverage of each UCC cluster by the subsample of this work
    UCC_cluster_coverage = np.full_like(UCC_cluster_sizes_full, 0.0, dtype=np.float64)
    valid = UCC_cluster_sizes_full > 0
    UCC_cluster_coverage[valid] = (
        UCC_cluster_sizes_in_subsample[valid] /
        UCC_cluster_sizes_full[valid]
    )
    del UCC_cluster_sizes_full  # Free memory
    gc.collect()  # Force garbage collection

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_UCC_members_cluster_ids_subsample, UCC_members_cluster_ids_subsample)
    np.save(file_path_UCC_members_cluster_membership_subsample, UCC_members_cluster_membership_subsample)
    np.save(file_path_UCC_cluster_sizes_in_subsample, UCC_cluster_sizes_in_subsample)
    np.save(file_path_UCC_cluster_coverage, UCC_cluster_coverage)
    del UCC_members_cluster_ids_subsample, UCC_members_cluster_membership_subsample, UCC_cluster_sizes_in_subsample, UCC_cluster_coverage  # Free memory
    gc.collect()  # Force garbage collection

def plot_UCC_clusters_on_sky(overwrite=False):
    """
    Plot the Unified Cluster Catalogue clusters on the sky.
    """
    # Check if plot already exists
    file_path_clusters_on_sky = os.path.join(OUTPUT_PATH, "UCC_clusters_on_sky.png")
    if os.path.exists(file_path_clusters_on_sky) and not overwrite:
        print(f"Unified Cluster Catalogue clusters plot already exists at:\n\t{file_path_clusters_on_sky} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Unified Cluster Catalogue clusters...")

    # Load the UCC clustering output
    print("... loading Unified Cluster Catalogue catalogue")
    UCC_members_cluster_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_members_cluster_ids_subsample.npy"))  # (N_subsample, max_appearances)
    UCC_members_cluster_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_members_cluster_membership_subsample.npy"))  # (N_subsample, max_appearances)

    # Plot the open clusters on the sky
    plot_catalogue_structure_on_sky(
        UCC_members_cluster_ids_subsample,
        UCC_members_cluster_membership_subsample,
        file_path_clusters_on_sky
    )

def compare_to_UCC(overwrite=False):
    """
    Compare the clustering output to the Unified Cluster Catalogue.
    """
    # Check if comparison results already exist
    file_path_catalogue_rpj_allS = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_catalogue_rpj_allS.npy")
    file_path_catalogue_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_catalogue_idx_optimalS.npy")
    file_path_astrolink_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_astrolink_rpj_optimalS.npy")
    file_path_astrolink_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_astrolink_idx_optimalS.npy")
    file_path_catalogue_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_catalogue_rpj_optimalS.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_catalogue_rpj_allS) and
                 os.path.exists(file_path_catalogue_idx_optimalS) and
                 os.path.exists(file_path_astrolink_rpj_optimalS) and
                 os.path.exists(file_path_astrolink_idx_optimalS) and
                 os.path.exists(file_path_catalogue_rpj_optimalS))
    if all_exist and not overwrite:
        print("Unified Cluster Catalogue comparison results already exist at:")
        print(f"\t{file_path_catalogue_rpj_allS} ,")
        print(f"\t{file_path_catalogue_idx_optimalS} ,")
        print(f"\t{file_path_astrolink_rpj_optimalS} ,")
        print(f"\t{file_path_astrolink_idx_optimalS} , and")
        print(f"\t{file_path_catalogue_rpj_optimalS} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to the Unified Cluster Catalogue...")

    # Load the reduced UCC data
    print("... loading reduced UCC data")
    UCC_members_cluster_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_members_cluster_ids_subsample.npy"))  # (N_subsample, max_appearances)
    UCC_members_cluster_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_members_cluster_membership_subsample.npy"))  # (N_subsample, max_appearances)
    UCC_cluster_sizes_in_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_cluster_sizes_in_subsample.npy"))  # (N_clusters,)

    # Choose comparison metric
    comparison_metric = 3  # 0 for recovery, 1 for purity (S_C=S_A), 2 for purity (S_C=all_catalogue_stars), 3 for Jaccard index (S_C=S_A), 4 for Jaccard index (S_C=all_catalogue_stars)
    
    # Make the comparison
    compare_to_catalogue_helper(
        UCC_members_cluster_ids_subsample,
        UCC_members_cluster_membership_subsample,
        UCC_cluster_sizes_in_subsample,
        comparison_metric,
        file_path_catalogue_rpj_allS,
        file_path_catalogue_idx_optimalS,
        file_path_astrolink_rpj_optimalS,
        file_path_astrolink_idx_optimalS,
        file_path_catalogue_rpj_optimalS
    )

def plot_UCC_crossmatch_per_significance(overwrite=False):
    """
    Plot a summary the comparison between the AstroLink clustering output and the 
    Unified Cluster Catalogue (UCC) as a function of the significance threshold S.
    """
    # Check if plot already exists
    file_path = os.path.join(OUTPUT_PATH, "UCC_crossmatch_per_significance.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Unified Cluster Catalogue comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Unified Cluster Catalogue comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJ = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_catalogue_rpj_allS.npy"))  # (N_sigmas, N_clusters, 5)
    UCC_cluster_coverage = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_cluster_coverage.npy"))

    # Load UCC cluster metadata
    print("... loading Unified Cluster Catalogue cluster names and quality classes")
    UCC_clusters_names = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_clusters_names.npy"))
    UCC_clusters_quality_class = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_clusters_quality_class.npy"))

    # Reorder the quality classes by the sorted names to match RPJ
    reorder = np.argsort(UCC_clusters_names)
    UCC_clusters_names = UCC_clusters_names[reorder]
    UCC_clusters_quality_class = UCC_clusters_quality_class[reorder]

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # ========== QUALITY CLASS GROUPS ==========
    print("... plotting statistics vs significance level for UCC quality ranges")
    cluster_class_lists = [
        ['AA', 'AB', 'BA'],
        ['AC', 'BB', 'CA'],
        ['AD', 'BC', 'CB', 'DA'],
        ['BD', 'CC', 'DB'],
        ['CD', 'DC', 'DD']
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("cluster_classes_cmap", [(0, 'C0'), (1, 'C3')])
    cluster_class_colours = [cmap(i / (len(cluster_class_lists) - 1)) for i in range(len(cluster_class_lists))]

    # Containers for legend handles
    handles_jaccard = []
    handles_matched = []
    labels = []

    for class_list, colour in zip(cluster_class_lists, cluster_class_colours):
        mask = np.zeros(RPJ.shape[1], dtype=bool)
        for qclass in class_list:
            mask |= (UCC_clusters_quality_class == qclass)
        if not np.any(mask):
            continue

        # String representation for labels
        class_list_string = f"{{{', '.join(class_list)}}}"

        # Extract Jaccard + evidence per assumption
        J_full, J_union = RPJ[:, mask, 3], RPJ[:, mask, 4]
        masked_coverage = UCC_cluster_coverage[mask]  # (N_clusters,)

        # Weighted means
        mask_coverage_sum = np.sum(masked_coverage)
        Jbar_full  = np.sum(J_full * masked_coverage, axis=1) / mask_coverage_sum
        Jbar_union = np.sum(J_union * masked_coverage, axis=1) / mask_coverage_sum

        # Plot both bounds + hatched region (Jaccard)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
                color=colour, linestyle='solid', linewidth=1, zorder=2)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
                color=colour, linestyle='solid', linewidth=1, zorder=2)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
                color=colour, alpha=0.3, zorder=2)

        # Matched fractions
        fraction_matched_union = np.mean(J_union > 0.5, axis=1)
        fraction_matched_full = np.mean(J_full > 0.5, axis=1)

        # Plot matched fractions (dashed)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_union,
                color=colour, linestyle='dashed', linewidth=1, zorder=2)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full,
                color=colour, linestyle='dashed', linewidth=1, zorder=2)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full, fraction_matched_union,
                facecolor='none', hatch='//', edgecolor=colour, linewidth=1, zorder=2)
    
    # Plot optimal S value on top
    ax.axvline(OPTIMAL_SIGMA_THRESHOLD, color='grey', linestyle='dotted', linewidth=0.75, zorder=4)
    ax.text(
        OPTIMAL_SIGMA_THRESHOLD, 1.01,
        f"S = {OPTIMAL_SIGMA_THRESHOLD}",
        color='grey', fontsize=10, ha='center', va='bottom', zorder=4
    )

    # ========== Final formatting ==========
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Significance, $S$")
    ax.set_ylabel("Comparison statistic")

    # Dummy handles for the legend
    handles = [
        Rectangle((0,0), 1, 1, facecolor=mcolors.to_rgba('k', alpha=0.3), edgecolor='k', linewidth=1, label='Jaccard index'),
        Rectangle((0,0), 1, 1, facecolor='none', hatch='//', edgecolor='k', linewidth=1, label='Matched fraction'),
        plt.Line2D([], [], color='none', label='AA, AB, BA'),
        plt.Line2D([], [], color='none', label='AC, BB, CA'),
        plt.Line2D([], [], color='none', label='AD, BC, CB, DA'),
        plt.Line2D([], [], color='none', label='BD, CC, DB'),
        plt.Line2D([], [], color='none', label='CD, DC, DD'),
    ]

    # Create the legend
    leg = ax.legend(handles=handles, loc='upper right', frameon=False)

    # Define cluster type colors
    cluster_type_colors = {
        'AA, AB, BA': cluster_class_colours[0],
        'AC, BB, CA': cluster_class_colours[1],
        'AD, BC, CB, DA': cluster_class_colours[2],
        'BD, CC, DB': cluster_class_colours[3],
        'CD, DC, DD': cluster_class_colours[4]
    }

    # Recolor legend text entries for cluster types
    for text in leg.get_texts():
        label = text.get_text()
        if label in cluster_type_colors:
            text.set_color(cluster_type_colors[label])
            text.set_fontweight('bold')

    print("... saving figure.\n")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Compare clustering output to Vasiliev & Baumgardt (2021) ===
def prepare_Vasiliev2021_for_comparison(overwrite=False):
    """
    Prepare the Vasiliev & Baumgardt (2021) catalogue for comparison to the clustering output.
    """
    # Check if Vasiliev2021 folder exists in INTERMEDIATE_FILES_PATH
    vasiliev2021_path = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021")
    os.makedirs(vasiliev2021_path, exist_ok=True)

    # Check if files already exist
    file_path_V21_clusters_names = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_clusters_names.npy")
    file_path_V21_members_cluster_ids_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_members_cluster_ids_subsample.npy")
    file_path_V21_members_cluster_membership_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_members_cluster_membership_subsample.npy")
    file_path_V21_cluster_sizes_in_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_cluster_sizes_in_subsample.npy")
    file_path_V21_cluster_coverage = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_cluster_coverage.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_V21_clusters_names) and
                 os.path.exists(file_path_V21_members_cluster_ids_subsample) and
                 os.path.exists(file_path_V21_members_cluster_membership_subsample) and
                 os.path.exists(file_path_V21_cluster_sizes_in_subsample) and
                 os.path.exists(file_path_V21_cluster_coverage))
    if all_exist and not overwrite:
        print("Vasiliev & Baumgardt (2021) reduced data already exists at:")
        print(f"\t{file_path_V21_clusters_names} ,")
        print(f"\t{file_path_V21_members_cluster_ids_subsample} ,")
        print(f"\t{file_path_V21_members_cluster_membership_subsample} ,")
        print(f"\t{file_path_V21_cluster_sizes_in_subsample} , and")
        print(f"\t{file_path_V21_cluster_coverage} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with the Vasiliev & Baumgardt (2021) catalogue...")

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Load required arrays
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N_gdr3,)

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

        # Save the cluster names
        V21_clusters_names = np.array([os.path.splitext(os.path.basename(name))[0] for name in globular_cluster_files])
        np.save(file_path_V21_clusters_names, V21_clusters_names)
        del V21_clusters_names  # Free memory
        gc.collect()  # Force garbage collection
        
        # Cycle through each cluster and save the star memberships for that cluster
        max_V21_cluster_ID = len(globular_cluster_files) - 1
        V21_members_cluster_ids_gdr3 = np.full((subsample_mask.size, 1), max_V21_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_V21_cluster_ID + 1, representing no cluster
        V21_members_cluster_membership_gdr3 = np.zeros((subsample_mask.size, 1), dtype=np.bool_)  # (N,) Initialize with 0, representing zero membership probability
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
            
            # Convert probabilistic membership to a binary one
            members_cluster_membership = cluster_member_probs >= 0.5

            # Find which stars in the full GDR3 catalogue are in this cluster
            indices = np.searchsorted(gdr3_source_ids, cluster_source_ids)  # Assumes gdr3_source_ids is sorted
            indices = indices[(indices < gdr3_source_ids.size) & (gdr3_source_ids[indices] == cluster_source_ids)]  # Keep only valid matches

            # Assign the cluster IDs and memberships to the members
            unassigned_bool = True
            cluster_source_ids_indices = V21_members_cluster_ids_gdr3[indices]
            for j in range(V21_members_cluster_ids_gdr3.shape[1]):
                # Find which members can be assigned to this column
                unassigned_indices_bool = cluster_source_ids_indices[:, j] == max_V21_cluster_ID + 1

                # Assign the cluster IDs and memberships to the unassigned members for this column
                unassigned_indices = indices[unassigned_indices_bool]
                V21_members_cluster_ids_gdr3[unassigned_indices, j] = i
                V21_members_cluster_membership_gdr3[unassigned_indices, j] = members_cluster_membership[unassigned_indices_bool]

                # Update the arrays to only include those that are still unassigned
                indices = indices[~unassigned_indices_bool]
                cluster_source_ids_indices = cluster_source_ids_indices[~unassigned_indices_bool]
                members_cluster_membership = members_cluster_membership[~unassigned_indices_bool]

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
                V21_members_cluster_membership_gdr3 = np.concatenate(
                    (V21_members_cluster_membership_gdr3, np.zeros((subsample_mask.size, 1), dtype=np.bool_)),
                    axis=1
                )

                # Assign the cluster IDs and memberships to the members
                V21_members_cluster_ids_gdr3[indices, -1] = i
                V21_members_cluster_membership_gdr3[indices, -1] = members_cluster_membership[indices]
    del gdr3_source_ids  # Free memory
    gc.collect()  # Force garbage collection
    
    # Get the cluster IDs and memberships for the subsample in this work
    V21_members_cluster_ids_subsample = V21_members_cluster_ids_gdr3[subsample_mask]
    V21_members_cluster_membership_subsample = V21_members_cluster_membership_gdr3[subsample_mask]
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total size for each globular cluster in Vasiliev & Baumgardt (2021)
    print("... pre-computing the total size for each Vasiliev & Baumgardt (2021) globular cluster")
    V21_cluster_sizes_full = np.bincount(
        V21_members_cluster_ids_gdr3.ravel(),
        weights=V21_members_cluster_membership_gdr3.ravel(),
        minlength=max_V21_cluster_ID + 1
    )[:max_V21_cluster_ID + 1]  # (N_clusters,)
    del V21_members_cluster_ids_gdr3, V21_members_cluster_membership_gdr3  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the size for each globular cluster in the overlap with the subsample
    print("... pre-computing the size for each Vasiliev & Baumgardt (2021) globular cluster in the subsample of this work")
    V21_cluster_sizes_in_subsample = np.bincount(
        V21_members_cluster_ids_subsample.ravel(),
        weights=V21_members_cluster_membership_subsample.ravel(),
        minlength=max_V21_cluster_ID + 1
    )[:max_V21_cluster_ID + 1]  # (N_clusters,)

    # Calculate the coverage of each globular cluster by the subsample of this work
    V21_cluster_coverage = np.full_like(V21_cluster_sizes_full, 0.0, dtype=np.float64)
    valid = V21_cluster_sizes_full > 0
    V21_cluster_coverage[valid] = (
        V21_cluster_sizes_in_subsample[valid] /
        V21_cluster_sizes_full[valid]
    )
    del V21_cluster_sizes_full  # Free memory
    gc.collect()  # Force garbage collection

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_V21_members_cluster_ids_subsample, V21_members_cluster_ids_subsample)
    np.save(file_path_V21_members_cluster_membership_subsample, V21_members_cluster_membership_subsample)
    np.save(file_path_V21_cluster_sizes_in_subsample, V21_cluster_sizes_in_subsample)
    np.save(file_path_V21_cluster_coverage, V21_cluster_coverage)
    del V21_members_cluster_ids_subsample, V21_members_cluster_membership_subsample, V21_cluster_sizes_in_subsample, V21_cluster_coverage  # Free memory
    gc.collect()  # Force garbage collection

def plot_Vasiliev2021_clusters_on_sky(overwrite=False):
    """
    Plot the Vasiliev & Baumgardt (2021) globular clusters on the sky.
    """
    # Check if plot already exists
    file_path_clusters_on_sky = os.path.join(OUTPUT_PATH, "Vasiliev2021_clusters_on_sky.png")
    if os.path.exists(file_path_clusters_on_sky) and not overwrite:
        print(f"Vasiliev & Baumgardt (2021) globular clusters on sky plot already exists at:\n\t{file_path_clusters_on_sky} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Vasiliev & Baumgardt (2021) globular clusters on the sky...")

    # Load the Vasiliev & Baumgardt (2021) clustering output
    print("... loading Vasiliev & Baumgardt (2021) clustering output for plotting")
    V21_members_cluster_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_members_cluster_ids_subsample.npy"))  # (N_subsample, max_appearances)
    V21_members_cluster_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_members_cluster_membership_subsample.npy"))  # (N_subsample, max_appearances)
    
    # Plot the globular clusters on the sky
    plot_catalogue_structure_on_sky(
        V21_members_cluster_ids_subsample,
        V21_members_cluster_membership_subsample,
        file_path_clusters_on_sky
    )

def compare_to_Vasiliev2021(overwrite=False):
    """
    Compare the clustering output to the Vasiliev & Baumgardt (2021) catalogue.
    """
    # Check if comparison results already exist
    file_path_catalogue_rpj_allS = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_catalogue_rpj_allS.npy")
    file_path_catalogue_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_catalogue_idx_optimalS.npy")
    file_path_astrolink_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_astrolink_rpj_optimalS.npy")
    file_path_astrolink_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_astrolink_idx_optimalS.npy")
    file_path_catalogue_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_catalogue_rpj_optimalS.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_catalogue_rpj_allS) and
                 os.path.exists(file_path_catalogue_idx_optimalS) and
                 os.path.exists(file_path_astrolink_rpj_optimalS) and
                 os.path.exists(file_path_astrolink_idx_optimalS) and
                 os.path.exists(file_path_catalogue_rpj_optimalS))
    if all_exist and not overwrite:
        print("Vasiliev & Baumgardt (2021) comparison results already exist at:")
        print(f"\t{file_path_catalogue_rpj_allS} ,")
        print(f"\t{file_path_catalogue_idx_optimalS} ,")
        print(f"\t{file_path_astrolink_rpj_optimalS} ,")
        print(f"\t{file_path_astrolink_idx_optimalS} , and")
        print(f"\t{file_path_catalogue_rpj_optimalS} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to the Vasiliev & Baumgardt (2021) catalogue...")

    # Load the reduced Vasiliev & Baumgardt (2021) data
    print("... loading reduced Vasiliev & Baumgardt (2021) data")
    V21_members_cluster_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_members_cluster_ids_subsample.npy"))  # (N_subsample, max_appearances)
    V21_members_cluster_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_members_cluster_membership_subsample.npy"))  # (N_subsample, max_appearances)
    V21_cluster_sizes_in_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_cluster_sizes_in_subsample.npy"))  # (N_clusters,)

    # Choose comparison metric
    comparison_metric = 4  # 0 for recovery, 1 for purity (S_C=S_A), 2 for purity (S_C=all_catalogue_stars), 3 for Jaccard index (S_C=S_A), 4 for Jaccard index (S_C=all_catalogue_stars)
    
    # Make the comparison
    compare_to_catalogue_helper(
        V21_members_cluster_ids_subsample,
        V21_members_cluster_membership_subsample,
        V21_cluster_sizes_in_subsample,
        comparison_metric,
        file_path_catalogue_rpj_allS,
        file_path_catalogue_idx_optimalS,
        file_path_astrolink_rpj_optimalS,
        file_path_astrolink_idx_optimalS,
        file_path_catalogue_rpj_optimalS
    )

def plot_Vasiliev2021_crossmatch_per_significance(overwrite=False):
    """
    Plot a summary the comparison between the AstroLink clustering output and the 
    Vasiliev & Baumgardt (2021) catalogue as a function of the significance threshold S.
    """
    # Check if plot already exists
    file_path = os.path.join(OUTPUT_PATH, "Vasiliev2021_crossmatch_per_significance.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Vasiliev & Baumgardt (2021) comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Vasiliev & Baumgardt (2021) comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJ = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_catalogue_rpj_allS.npy"))  # (N_sigmas, N_clusters, 5)
    V21_cluster_coverage = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_cluster_coverage.npy"))

    # Load cluster names
    V21_clusters_names = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_clusters_names.npy"))  # (N_clusters,)

    # Extract per-assumption and per-statistic arrays
    J_full,  J_union  = RPJ[:, :, 3], RPJ[:, :, 4]

    # Coverage-weighted averages
    coverage_sum = np.sum(V21_cluster_coverage)
    Jbar_full  = np.sum(J_full * V21_cluster_coverage, axis=1) / coverage_sum
    Jbar_union = np.sum(J_union * V21_cluster_coverage, axis=1) / coverage_sum

    print("... plotting combined statistics vs significance level")

    # Define clusters and colors
    best_matching_clusters = np.argsort(np.max(J_union, axis=0))[::-1][:5]
    cluster_type_colors = {
        cluster_name: f"C{index}"
        for index, cluster_name in enumerate(V21_clusters_names[best_matching_clusters])
    }

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
            color='k', linestyle='solid', linewidth=1, zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
            color='k', linestyle='solid', linewidth=1, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
            color='k', alpha=0.3, zorder=3)

    # Cycle through specific clusters and plot their Jaccard indices vs significance level
    for cluster_name, color in cluster_type_colors.items():
        # Find index of this cluster
        cluster_index = np.where(V21_clusters_names == cluster_name)[0][0]

        # Plot Jaccard indices for this cluster
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, J_union[:, cluster_index],
                color=color, linestyle='solid', linewidth=1, zorder=4)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, J_full[:, cluster_index],
                color=color, linestyle='solid', linewidth=1, zorder=4)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, J_full[:, cluster_index], J_union[:, cluster_index],
                color=color, alpha=0.3, zorder=4)
    
    # Matched fractions
    fraction_matched_union = np.mean(J_union > 0.5, axis=1)
    fraction_matched_full = np.mean(J_full > 0.5, axis=1)

    # Plot fraction of clusters of this type matched to above J=0.5
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_union,
            color='k', linestyle='dashed', linewidth=1, zorder=2)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full,
            color='k', linestyle='dashed', linewidth=1, zorder=2)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full, fraction_matched_union,
            facecolor='none', hatch='//', edgecolor='k', linewidth=1, zorder=2)

    # Plot optimal S value on top
    ax.axvline(OPTIMAL_SIGMA_THRESHOLD, color='grey', linestyle='dotted', linewidth=0.75, zorder=4)
    ax.text(
        OPTIMAL_SIGMA_THRESHOLD, 1.01,
        f"S = {OPTIMAL_SIGMA_THRESHOLD}",
        color='grey', fontsize=10, ha='center', va='bottom', zorder=4
    )

    # Dummy handles for the legend
    handles = [
        Rectangle((0,0), 1, 1, facecolor=mcolors.to_rgba('k', alpha=0.3), edgecolor='k', linewidth=1, label='Jaccard index'),
        Rectangle((0,0), 1, 1, facecolor='none', hatch='//', edgecolor='k', linewidth=1, label='Matched fraction'),
    ] + [plt.Line2D([], [], color='none', label=cluster_name) for cluster_name in cluster_type_colors.keys()]

    # Create the legend
    leg = ax.legend(
        handles=handles,
        loc='lower left',
        bbox_to_anchor=(0.375, 0),
        frameon=False
    )

    # Recolor legend text entries for cluster types
    for text in leg.get_texts():
        label = text.get_text()
        if label in cluster_type_colors:
            text.set_color(cluster_type_colors[label])
            text.set_fontweight('bold')

    # Final formatting
    print("... saving figure.\n")
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Significance, $S$")
    ax.set_ylabel("Comparison statistic")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Compare clustering output to Battaglia et al. 2021 catalogue ===
def prepare_Battaglia2021_for_comparison(overwrite=False):
    """
    Prepare the Battaglia et al. (2021) catalogue for comparison to the clustering output.
    """
    # Check if Battaglia2021 folder exists in INTERMEDIATE_FILES_PATH
    battaglia2021_path = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021")
    os.makedirs(battaglia2021_path, exist_ok=True)

    # Check if files already exist
    file_path_dwarfgalaxies_names = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_dwarfgalaxies_names.npy")
    file_path_B21_members_dwarfgalaxy_ids_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_members_dwarfgalaxy_ids_subsample.npy")
    file_path_B21_members_dwarfgalaxy_membership_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_members_dwarfgalaxy_membership_subsample.npy")
    file_path_B21_dwarfgalaxy_sizes_in_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_dwarfgalaxy_sizes_in_subsample.npy")
    file_path_B21_dwarfgalaxy_coverage = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_dwarfgalaxy_coverage.npy")
    
    # Skip if all files exist and overwrite is False
    all_exist = (os.path.exists(file_path_dwarfgalaxies_names) and
                 os.path.exists(file_path_B21_members_dwarfgalaxy_ids_subsample) and
                 os.path.exists(file_path_B21_members_dwarfgalaxy_membership_subsample) and
                 os.path.exists(file_path_B21_dwarfgalaxy_sizes_in_subsample) and
                 os.path.exists(file_path_B21_dwarfgalaxy_coverage))
    if all_exist and not overwrite:
        print("Battaglia et al. (2021) reduced data already exists at:")
        print(f"\t{file_path_dwarfgalaxies_names} ,")
        print(f"\t{file_path_B21_members_dwarfgalaxy_ids_subsample} ,")
        print(f"\t{file_path_B21_members_dwarfgalaxy_membership_subsample} ,")
        print(f"\t{file_path_B21_dwarfgalaxy_sizes_in_subsample} , and")
        print(f"\t{file_path_B21_dwarfgalaxy_coverage} .")
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

    # Convert probabilistic membership to a binary one
    B21_members_dwarfgalaxy_membership = B21_members_probs >= 0.5

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Load subsample mask
    print("... loading subsample mask")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))  # (N,)

    # Save the membership mask
    print("... making membership mask")
    indices = np.searchsorted(gdr3_source_ids, B21_members_source_ids) # Assumes gdr3_source_ids is sorted
    B21_members_mask = np.zeros_like(gdr3_source_ids, dtype=np.bool_)  # Create a mask of the same shape as gdr3_source_ids
    B21_members_mask[indices] = True  # Set the indices of the members to True
    del gdr3_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Get Battaglia+2021 dwarf galaxy IDs for this subsample (multiple columns since stars can be in multiple Battaglia+2021 dwarf galaxies)
    print("... getting Battaglia+2024 dwarf galaxy IDs and memberships for the subsample in this work")
    max_appearances = np.unique(B21_members_source_ids, return_counts=True)[1].max()
    max_B21_dwarfgalaxy_ID = B21_members_dwarfgalaxy_ids.max()
    B21_members_dwarfgalaxy_ids_gdr3 = np.full((B21_members_mask.size, max_appearances), max_B21_dwarfgalaxy_ID + 1, dtype=np.int64)  # (N,) Initialize with max_B21_dwarfgalaxy_ID + 1, representing no cluster
    B21_members_dwarfgalaxy_membership_gdr3 = np.zeros((B21_members_mask.size, max_appearances), dtype=np.bool_)  # (N,) Initialize with 0, representing zero membership probability
    B21_members_mask_where = np.where(B21_members_mask)[0]  # Use indices of members in the full catalogue from now on to do efficient slicing/indexing
    del B21_members_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Assign dwarf galaxy IDs / memberships to each of the members
    unique_B21_source_ids, indices, counts = np.unique(B21_members_source_ids, return_index=True, return_counts=True)
    for num_count in range(1, max_appearances + 1):
        # Get the relative position of the members in the full catalogue for this count
        num_count_bool = counts == num_count

        # Get the dwarf galaxy IDs and memberships for the members with this count
        if num_count == 1:
            dwarfgalaxy_ids = B21_members_dwarfgalaxy_ids[indices[num_count_bool]][:, None]
            membership = B21_members_dwarfgalaxy_membership[indices[num_count_bool]][:, None]
        else: # There are not too many of these so what follows is efficient enough
            dwarfgalaxy_ids = np.zeros((num_count_bool.sum(), num_count), dtype=np.int64)
            membership = np.zeros((num_count_bool.sum(), num_count), dtype=np.bool_)
            for i, sid in enumerate(unique_B21_source_ids[num_count_bool]):
                # Get the indices of the members with this source ID
                source_id_match = np.where(B21_members_source_ids == sid)[0]

                # Get the dwarf galaxy IDs and memberships for these members
                dwarfgalaxy_ids[i] = B21_members_dwarfgalaxy_ids[source_id_match]
                membership[i] = B21_members_dwarfgalaxy_membership[source_id_match]

        # Assign the dwarf galaxy IDs and memberships to the members
        B21_members_dwarfgalaxy_ids_gdr3[B21_members_mask_where[num_count_bool], :num_count] = dwarfgalaxy_ids
        B21_members_dwarfgalaxy_membership_gdr3[B21_members_mask_where[num_count_bool], :num_count] = membership

    B21_members_dwarfgalaxy_ids_subsample = B21_members_dwarfgalaxy_ids_gdr3[subsample_mask]
    B21_members_dwarfgalaxy_membership_subsample = B21_members_dwarfgalaxy_membership_gdr3[subsample_mask]
    del subsample_mask, B21_members_source_ids, B21_members_dwarfgalaxy_ids_gdr3, B21_members_dwarfgalaxy_membership_gdr3, B21_members_mask_where, unique_B21_source_ids, indices, counts  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total size for each Battaglia+2021 dwarf galaxy
    print("... pre-computing the total size for each Battaglia+2021 dwarf galaxy")
    B21_dwarfgalaxy_sizes_full = np.bincount(
        B21_members_dwarfgalaxy_ids.ravel(),
        weights=B21_members_dwarfgalaxy_membership.ravel(),
        minlength=max_B21_dwarfgalaxy_ID + 1
    )  # (N_dwarfgalaxies,)
    del B21_members_dwarfgalaxy_ids, B21_members_dwarfgalaxy_membership  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the size for each Battaglia+2021 dwarf galaxy in the overlap with the subsample
    print("... pre-computing the size for each Battaglia+2021 dwarf galaxy in the subsample of this work")
    B21_dwarfgalaxy_sizes_in_subsample = np.bincount(
        B21_members_dwarfgalaxy_ids_subsample.ravel(),
        weights=B21_members_dwarfgalaxy_membership_subsample.ravel(),
        minlength=max_B21_dwarfgalaxy_ID + 1
    )[:max_B21_dwarfgalaxy_ID + 1]  # (N_dwarfgalaxies,)

    # Calculate the coverage of each Battaglia+2021 dwarf galaxy by the subsample of this work
    B21_dwarfgalaxy_coverage = np.full_like(B21_dwarfgalaxy_sizes_full, 0.0, dtype=np.float64)
    valid = B21_dwarfgalaxy_sizes_full > 0
    B21_dwarfgalaxy_coverage[valid] = (
        B21_dwarfgalaxy_sizes_in_subsample[valid] /
        B21_dwarfgalaxy_sizes_full[valid]
    )
    del B21_dwarfgalaxy_sizes_full  # Free memory
    gc.collect()  # Force garbage collection

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_B21_members_dwarfgalaxy_ids_subsample, B21_members_dwarfgalaxy_ids_subsample)
    np.save(file_path_B21_members_dwarfgalaxy_membership_subsample, B21_members_dwarfgalaxy_membership_subsample)
    np.save(file_path_B21_dwarfgalaxy_sizes_in_subsample, B21_dwarfgalaxy_sizes_in_subsample)
    np.save(file_path_B21_dwarfgalaxy_coverage, B21_dwarfgalaxy_coverage)
    del B21_members_dwarfgalaxy_ids_subsample, B21_members_dwarfgalaxy_membership_subsample, B21_dwarfgalaxy_sizes_in_subsample, B21_dwarfgalaxy_coverage  # Free memory
    gc.collect()  # Force garbage collection

def plot_Battaglia2021_dwarfgalaxies_on_sky(overwrite=False):
    """
    Plot the Battaglia et al. (2021) dwarf galaxies on the sky.
    """
    # Check if plot already exists
    file_path_dwarfgalaxies_on_sky = os.path.join(OUTPUT_PATH, "Battaglia2021_dwarfgalaxies_on_sky.png")
    if os.path.exists(file_path_dwarfgalaxies_on_sky) and not overwrite:
        print(f"Battaglia et al. (2021) dwarf galaxies on sky plot already exists at:\n\t{file_path_dwarfgalaxies_on_sky} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Battaglia et al. (2021) dwarf galaxies on the sky...")

    # Load the Battaglia et al. (2021) clustering output
    print("... loading Battaglia et al. (2021) clustering output for plotting")
    B21_members_dwarfgalaxy_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_members_dwarfgalaxy_ids_subsample.npy"))  # (N_subsample, max_appearances)
    B21_members_dwarfgalaxy_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_members_dwarfgalaxy_membership_subsample.npy"))  # (N_subsample, max_appearances)
    
    # Plot the dwarf galaxies on the sky
    plot_catalogue_structure_on_sky(
        B21_members_dwarfgalaxy_ids_subsample,
        B21_members_dwarfgalaxy_membership_subsample,
        file_path_dwarfgalaxies_on_sky
    )

def compare_to_Battaglia2021(overwrite=False):
    """
    Compare the clustering output to the Battaglia et al. (2021) catalogue.
    """
    # Check if comparison results already exist
    file_path_catalogue_rpj_allS = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_catalogue_rpj_allS.npy")
    file_path_catalogue_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_catalogue_idx_optimalS.npy")
    file_path_astrolink_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_astrolink_rpj_optimalS.npy")
    file_path_astrolink_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_astrolink_idx_optimalS.npy")
    file_path_catalogue_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_catalogue_rpj_optimalS.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_catalogue_rpj_allS) and
                 os.path.exists(file_path_catalogue_idx_optimalS) and
                 os.path.exists(file_path_astrolink_rpj_optimalS) and
                 os.path.exists(file_path_astrolink_idx_optimalS) and
                 os.path.exists(file_path_catalogue_rpj_optimalS))
    if all_exist and not overwrite:
        print("Battaglia et al. (2021) comparison results already exist at:")
        print(f"\t{file_path_catalogue_rpj_allS} ,")
        print(f"\t{file_path_catalogue_idx_optimalS} ,")
        print(f"\t{file_path_astrolink_rpj_optimalS} ,")
        print(f"\t{file_path_astrolink_idx_optimalS} , and")
        print(f"\t{file_path_catalogue_rpj_optimalS} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to Battaglia et al. (2021)...")

    # Load required arrays
    print("... loading required arrays for comparison")
    B21_members_dwarfgalaxy_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_members_dwarfgalaxy_ids_subsample.npy"))  # (N_subsample, max_appearances)
    B21_members_dwarfgalaxy_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_members_dwarfgalaxy_membership_subsample.npy"))  # (N_subsample, max_appearances)
    B21_dwarfgalaxy_sizes_in_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_dwarfgalaxy_sizes_in_subsample.npy")) # (N_dwarfgalaxies,)

    # Choose comparison metric
    comparison_metric = 4  # 0 for recovery, 1 for purity (S_C=S_A), 2 for purity (S_C=all_catalogue_stars), 3 for Jaccard index (S_C=S_A), 4 for Jaccard index (S_C=all_catalogue_stars)
    
    # Make the comparison
    compare_to_catalogue_helper(
        B21_members_dwarfgalaxy_ids_subsample,
        B21_members_dwarfgalaxy_membership_subsample,
        B21_dwarfgalaxy_sizes_in_subsample,
        comparison_metric,
        file_path_catalogue_rpj_allS,
        file_path_catalogue_idx_optimalS,
        file_path_astrolink_rpj_optimalS,
        file_path_astrolink_idx_optimalS,
        file_path_catalogue_rpj_optimalS
    )

def plot_Battaglia2021_crossmatch_per_significance(overwrite=False):
    """
    Plot a summary the comparison between the AstroLink clustering output and the 
    Battaglia et al. (2021) catalogue as a function of the significance threshold S.
    """
    # Check if plot already exists
    file_path = os.path.join(OUTPUT_PATH, "Battaglia2021_crossmatch_per_significance.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Battaglia et al. (2021) comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Battaglia et al. (2021) comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJ = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_catalogue_rpj_allS.npy"))  # (N_sigmas, N_dwarfgalaxies, 5)
    B21_cluster_coverage = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_dwarfgalaxy_coverage.npy"))

    # Load dwarf galaxy names
    dwarf_galaxy_names = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_dwarfgalaxies_names.npy"), allow_pickle=True)

    # Extract per-assumption and per-statistic arrays
    J_full,  J_union  = RPJ[:, :, 3], RPJ[:, :, 4]

    # Coverage-weighted averages
    coverage_sum = np.sum(B21_cluster_coverage)
    Jbar_full  = np.sum(J_full * B21_cluster_coverage, axis=1) / coverage_sum
    Jbar_union = np.sum(J_union * B21_cluster_coverage, axis=1) / coverage_sum

    print("... plotting combined statistics vs significance level")

    # Define dwarf galaxy names and colors
    best_matching_dwarf_galaxies = np.argsort(np.max(J_union, axis=0))[::-1][:5]
    dwarf_galaxy_colors = {
        dwarf_galaxy_name: f"C{index}"
        for index, dwarf_galaxy_name in enumerate(dwarf_galaxy_names[best_matching_dwarf_galaxies])
    }

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_union,
            color='k', linestyle='solid', linewidth=1, zorder=2)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full,
            color='k', linestyle='solid', linewidth=1, zorder=2)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Jbar_full, Jbar_union,
            color='k', alpha=0.3, zorder=3)

    # Cycle through specific streams and plot their Jaccard indices vs significance level
    for dwarf_galaxy_name, color in dwarf_galaxy_colors.items():
        # Find index of this stream
        dwarf_galaxy_index = np.where(dwarf_galaxy_names == dwarf_galaxy_name)[0][0]

        # Plot Jaccard indices for this stream
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, J_union[:, dwarf_galaxy_index],
                color=color, linestyle='solid', linewidth=1, zorder=3)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, J_full[:, dwarf_galaxy_index],
                color=color, linestyle='solid', linewidth=1, zorder=3)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, J_full[:, dwarf_galaxy_index], J_union[:, dwarf_galaxy_index],
                color=color, alpha=0.3, zorder=3)
    
    # Matched fractions
    fraction_matched_union = np.mean(J_union > 0.5, axis=1)
    fraction_matched_full = np.mean(J_full > 0.5, axis=1)

    # Plot fraction of clusters of this type matched to above J=0.5
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_union,
            color='k', linestyle='dashed', linewidth=1, zorder=2)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full,
            color='k', linestyle='dashed', linewidth=1, zorder=2)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full, fraction_matched_union,
            facecolor='none', hatch='//', edgecolor='k', linewidth=1, zorder=2)

    # Plot optimal S value on top
    ax.axvline(OPTIMAL_SIGMA_THRESHOLD, color='grey', linestyle='dotted', linewidth=0.75, zorder=4)
    ax.text(
        OPTIMAL_SIGMA_THRESHOLD, 1.01,
        f"S = {OPTIMAL_SIGMA_THRESHOLD}",
        color='grey', fontsize=10, ha='center', va='bottom', zorder=4
    )

    # Dummy handles for the legend
    handles = [
        Rectangle((0,0), 1, 1, facecolor=mcolors.to_rgba('k', alpha=0.3), edgecolor='k', linewidth=1, label='Jaccard index'),
        Rectangle((0,0), 1, 1, facecolor='none', hatch='//', edgecolor='k', linewidth=1, label='Matched fraction'),
    ] + [plt.Line2D([], [], color='none', label=dwarf_galaxy_name) for dwarf_galaxy_name in dwarf_galaxy_colors.keys()]

    # Create the legend
    leg = ax.legend(
        handles=handles,
        loc='center',
        bbox_to_anchor=(0.625, 0.4),
        frameon=False
    )

    # Recolor legend text entries for cluster types
    for text in leg.get_texts():
        label = text.get_text()
        if label in dwarf_galaxy_colors:
            text.set_color(dwarf_galaxy_colors[label])
            text.set_fontweight('bold')

    # Final formatting
    print("... saving figure.\n")
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Significance, $S$")
    ax.set_ylabel("Comparison statistic")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Compare clustering output to galstreams ===
def prepare_galstreams_for_comparison(overwrite=False):
    """
    Prepare the data for comparison with the galstreams catalogue.
    """
    # Check if galstreams folder exists in INTERMEDIATE_FILES_PATH
    galstreams_path = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams")
    os.makedirs(galstreams_path, exist_ok=True)
    
    # Check if files already exist
    file_path_galstreams_stream_track_names = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_stream_track_names.npy")
    file_path_galstreams_members_stream_ids_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_members_stream_ids_subsample.npy")
    file_path_galstreams_members_stream_membership_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_members_stream_membership_subsample.npy")
    file_path_galstreams_stream_sizes_in_subsample = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_stream_sizes_in_subsample.npy")
    file_path_galstreams_stream_coverage = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_stream_coverage.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_galstreams_stream_track_names) and
                 os.path.exists(file_path_galstreams_members_stream_ids_subsample) and
                 os.path.exists(file_path_galstreams_members_stream_membership_subsample) and
                 os.path.exists(file_path_galstreams_stream_sizes_in_subsample) and
                 os.path.exists(file_path_galstreams_stream_coverage))
    if all_exist and not overwrite:
        print("Galstreams reduced data already exists at:")
        print(f"\t{file_path_galstreams_stream_track_names} ,")
        print(f"\t{file_path_galstreams_members_stream_ids_subsample} ,")
        print(f"\t{file_path_galstreams_members_stream_membership_subsample} ,")
        print(f"\t{file_path_galstreams_stream_sizes_in_subsample} , and")
        print(f"\t{file_path_galstreams_stream_coverage} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with the galstreams catalogue...")

    # Load required arrays
    print('... loading required arrays')
    source_ids = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_source_ids.npy"))  # (N_gdr3,)
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, 'subsample_mask.npy'))  # (N_gdr3,)
    ra, dec = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_equatorial_coordinates.npy")).T  # Each (N_gdr3,) in degrees
    cos_dec = np.cos(np.deg2rad(dec))
    mu_ra, mu_dec = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_proper_motions.npy")).T  # Each (N_gdr3,) in mas/yr

    # Get MWStreams object from galstreams
    print('... creating MWStreams object')
    with printout_suppressor():  # Suppress the printout from galstreams
        mws = galstreams.MWStreams(print_topcat_friendly_files=False)

    # Save an array of stream track names
    print('... saving stream track names')
    stream_track_names = np.array(list(mws.keys()), dtype=np.str_)
    np.save(file_path_galstreams_stream_track_names, stream_track_names)
    del stream_track_names  # Free memory
    gc.collect()  # Force garbage collection

    # Manual fixes for missing width_phi2 values from galstreams
    mws.summary.loc['M30-S20', 'width_phi2'] = 0.10992290189439437 # From galstreams/tracks/track.st.M30.sollima2020.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['M5-G19', 'width_phi2'] = 0.00059498941000224  # From galstreams/tracks/track.st.M5.grillmair2019.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['NGC5053-L06', 'width_phi2'] = 0.5  # Based on Lauchner et al. (2006) maps and the cluster’s distance (~17.4 kpc), an angular width of ~0.5 deg is used a reasonable proxy.
    mws.summary.loc['NGC6362-S20', 'width_phi2'] = 0.03279231365173159 # From galstreams/tracks/track.st.NGC6362.sollima2020.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['Orphan-K23', 'width_phi2'] = 0.5610732369318953 # From galstreams/tracks/track.st.Orphan-Chenab.ibata2021.summary.ecsv (default track summary file has ~0 deg width)
    mws.summary.loc['Pal5-PW19', 'width_phi2'] = 0.30188212290570887 # From galstreams/tracks/track.st.Pal5.ibata2024.summary.ecsv (default track summary file has ~0 deg width)
    mws.summary.loc['Parallel-W18', 'width_phi2'] = 2  # Reasonable value considering Weiss et al. (2018) reported physical dispersions sigma (kpc) from SDSS fits
    mws.summary.loc['Perpendicular-W18', 'width_phi2'] = 2  # Reasonable value considering Weiss et al. (2018) reported physical dispersions sigma (kpc) from SDSS fits

    # Manual fixes for missing width_pm_phi1_cosphi2 and width_pm_phi2 values from galstreams
    mws.summary.loc['ACS-R21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.051429218098668614, 0.060361599720383706]  # From galstreams/tracks/track.st.ACS.ramos2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['Gaia-10-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.25549095266698924, 0.21102160186354021]  # From galstreams/tracks/track.st.Gaia-10.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Gaia-12-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.31870038997481237, 0.39533356136581626]  # From galstreams/tracks/track.st.Gaia-12.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Gaia-6-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.4852100507178606, 0.3356251329783215]  # From galstreams/tracks/track.st.Gaia-6.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Gaia-7-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.3693734621270431, 0.30817705138525747]  # From galstreams/tracks/track.st.Gaia-7.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Hrid-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.8130582153516146, 0.5597190294200691]  # From galstreams/tracks/track.st.Hrid.ibata2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['Jhelum-a-B19', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.08724815957044693, 0.17890077278048597]  # From galstreams/tracks/track.st.Jhelum-a.shipp2019.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Jhelum-b-B19', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.12468187056175481, 0.12024080502075932]  # From galstreams/tracks/track.st.Jhelum-b.shipp2019.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['M2-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.11258154325759534, 0.09524201623728742]  # From galstreams/tracks/track.st.M2.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Monoceros-R21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.024488308491996437, 0.025432522875930966]  # From galstreams/tracks/track.st.Monoceros.ramos2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['NGC1261-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.07311649510540366, 0.009119289396829662] # From galstreams/tracks/track.st.NGC1261.ibata2021.summary.ecsv (not read in automatically for some reason)
    mws.summary.loc['NGC1851-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.15733960159151045, 0.26652779597474907]  # From galstreams/tracks/track.st.NGC1851.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['NGC2298-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.3019327576574179, 0.28430356991838157]  # From galstreams/tracks/track.st.NGC2298.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['NGC288-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.37797158611441367, 0.2383575166183864]  # From galstreams/tracks/track.st.NGC288.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['OmegaCen-I21', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [1.3122549027325625, 1.7837014357475944]  # From galstreams/tracks/track.st.OmegaCen-Fimbulthul.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Orphan-K23', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.26468901721952365, 0.3147176316063631]  # From galstreams/tracks/track.st.Orphan-Chenab.ibata2021.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Pal5-PW19', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.15042483606050822, 0.16169990020590352]  # From galstreams/tracks/track.st.Pal5.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Ravi-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['SGP-S-Y22', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.24098786032510924, 0.42219519260381305]  # From galstreams/tracks/track.st.SGP-S.ibata2024.summary.ecsv (different from default track summary file which show small values)
    mws.summary.loc['Turbio-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['Wambelong-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['Willka_Yaku-S18', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors
    mws.summary.loc['Yangtze-Y23', ['width_pm_phi1_cosphi2', 'width_pm_phi2']] = [0.2, 0.2]  # Heuristic values based on typical proper motion errors

    # Define HEALPix map and pixel radius for footprint calculations
    level = 12
    nside = 2**level
    npix = hp.nside2npix(nside)
    pixel_indices = np.arange(npix)
    pixel_area = 4 * np.pi / (12 * nside**2)
    pixel_radius = np.rad2deg(np.sqrt(pixel_area / np.pi))  # In degrees

    # Find HEALPix pixel centres
    theta, pixel_ra = hp.pix2ang(nside, pixel_indices, nest=True)
    pixel_dec = np.pi / 2 - theta  # Convert theta to dec
    cos_pixel_dec = np.cos(pixel_dec)
    xyz_pixel_centres = np.column_stack([
        cos_pixel_dec * np.cos(pixel_ra),
        cos_pixel_dec * np.sin(pixel_ra),
        np.sin(pixel_dec)
    ])
    del theta, pixel_ra, pixel_dec, cos_pixel_dec  # Free memory
    gc.collect()  # Force garbage collection

    # Get HEALPix pixel indices for all stars at the chosen level
    all_stars_pixels = (source_ids >> 35) >> (2 * (12 - level))

    # Cycle through each stream, find which stars are in its footprint, and then compute potential stream membership with the information available
    max_galstream_stream_ID = len(mws) - 1
    galstreams_members_stream_ids_gdr3 = np.full((subsample_mask.size, 1), max_galstream_stream_ID + 1, dtype=np.int64)  # (N,) Initialize with max_galstreams_stream_ID + 1, representing no stream
    galstreams_members_stream_membership_gdr3 = np.zeros((subsample_mask.size, 1), dtype=np.bool_)  # (N,) Initialize with 0, representing zero membership probability
    for stream_id, (stream_track_name, stream) in enumerate(mws.items()):
        print(f'... processing stream {stream_id + 1}/{len(mws)} ({stream_track_name}) | defining stream footprint                                                 ', end='\r')
        # Width of the stream (in degrees) in the phi2 direction (perpendicular to the stream track)
        width_phi2 = mws.summary.loc[stream_track_name, 'width_phi2']

        # Effective angular selection radius (stream half-width + pixel radius buffer)
        angular_radius = np.deg2rad(width_phi2 / 2 + pixel_radius)

        # Convert stream track nodes to 3D unit vectors
        theta = np.deg2rad(90.0 - stream.track.dec.deg)
        phi = np.deg2rad(stream.track.ra.deg)
        stream_vectors = hp.ang2vec(theta, phi)
        del theta, phi  # Free memory
        gc.collect()  # Force garbage collection

        # Interpolate stream track to avoid gaps along phi1 direction (angular spacing between nodes must be much smaller than stream width)
        dot_products = np.sum(stream_vectors[:-1] * stream_vectors[1:], axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        segment_angles = np.arccos(dot_products)  # Compute angular separation between consecutive stream nodes (in radians)
        max_step = np.deg2rad(pixel_radius) # Maximum allowed step size (conservative choice)

        # Interpolate between stream nodes where segment angles exceed max_step, using spherical linear interpolation (SLERP)
        resampled_vectors = []
        for i in range(len(stream_vectors) - 1):
            v1 = stream_vectors[i]
            v2 = stream_vectors[i + 1]
            angle = segment_angles[i]

            # Determine number of interpolation steps required
            n_steps = max(1, int(np.ceil(angle / max_step)))

            if angle == 0.0:
                resampled_vectors.append(v1)
                continue

            sin_angle = np.sin(angle)

            # Spherical linear interpolation (SLERP)
            for t in np.linspace(0.0, 1.0, n_steps, endpoint=False):
                vec = (
                    np.sin((1.0 - t) * angle) / sin_angle * v1 +
                    np.sin(t * angle) / sin_angle * v2
                )
                resampled_vectors.append(vec)

        # Append final stream node
        resampled_vectors.append(stream_vectors[-1])
        resampled_vectors = np.array(resampled_vectors)
        del stream_vectors, dot_products, segment_angles  # Free memory
        gc.collect()  # Force garbage collection

        # Query HEALPix to find all pixels within angular_radius of resampled nodes
        selected_pixel_set = set()
        for vec in resampled_vectors:
            nearby_pixels = hp.query_disc(
                nside,
                vec,
                angular_radius,
                nest=True
            )
            selected_pixel_set.update(nearby_pixels)

        # Convert selected pixels to numpy array
        selected_pixels = np.array(list(selected_pixel_set), dtype=np.int64)
        del resampled_vectors, selected_pixel_set  # Free memory
        gc.collect()  # Force garbage collection

        # Make mask for which stars are in the stream footprint by doing a binary search for membership (faster and more memory efficient than np.isin)
        pixel_mask = np.zeros(npix, dtype=np.bool_)
        pixel_mask[selected_pixels] = True
        in_footprint_mask = pixel_mask[all_stars_pixels]
        del selected_pixels, pixel_mask  # Free memory
        gc.collect()  # Force garbage collection

        # Make SkyCoord object for stars in stream footprint
        print(f'... processing stream {stream_id + 1}/{len(mws)} ({stream_track_name}) | getting GDR3 stars in stream footprint        ', end='\r')
        dec_in_footprint = dec[in_footprint_mask]
        footprint_stars = SkyCoord(
            ra=ra[in_footprint_mask] * units.deg,
            dec=dec_in_footprint * units.deg,
            pm_ra_cosdec=mu_ra[in_footprint_mask] * cos_dec[in_footprint_mask] * units.mas / units.yr,
            pm_dec=mu_dec[in_footprint_mask] * units.mas / units.yr,
            frame='icrs'
        )
        del dec_in_footprint  # Free memory
        gc.collect()  # Force garbage collection

        # Transform footprint stars and stream_track into stream coordinates
        print(f'... processing stream {stream_id + 1}/{len(mws)} ({stream_track_name}) | transforming stars into stream coordinates    ', end='\r')
        footprint_stars = footprint_stars.transform_to(stream.stream_frame)
        stream.track = stream.track.transform_to(stream.stream_frame)

        # Extract phi1 and phi2 of footprint stars
        footprint_stars_phi1 = footprint_stars.phi1.deg
        footprint_stars_phi2 = footprint_stars.phi2.deg

        # Extract phi2 and phi1 of stream track
        track_phi1 = stream.track.phi1.deg
        track_phi2 = stream.track.phi2.deg

        # Interpolate track phi2 vs phi1
        interp_phi2 = np.interp(footprint_stars_phi1, track_phi1, track_phi2)

        # Distance in phi2 direction
        phi2_separation = np.abs(footprint_stars_phi2 - interp_phi2)
        del footprint_stars_phi2, track_phi2, interp_phi2

        # Further refine footprint stars into stream stars
        print(f'... processing stream {stream_id + 1}/{len(mws)} ({stream_track_name}) | separating stream and background stars        ', end='\r')
        in_stream_mask = phi2_separation < width_phi2 / 2      # Stream stars are inside +/- width_phi2 / 2
        in_stream_mask[(footprint_stars_phi1 < track_phi1.min()) | (footprint_stars_phi1 > track_phi1.max())] = False  # Remove stars outside track phi1 range from stream
        del footprint_stars_phi1  # Free memory
        gc.collect()  # Force garbage collection
        
        # Get stars in stream
        stream_stars = footprint_stars[in_stream_mask]
        del footprint_stars  # Free memory
        gc.collect()  # Force garbage collection

        # Proper motions (if available for stream and stars)
        if mws.summary.loc[stream_track_name, 'has_pm']:
            try:
                print(f'... processing stream {stream_id + 1}/{len(mws)} ({stream_track_name}) | refining stream stars with proper motions     ', end='\r')
                # Widths in proper motions
                width_pm1, width_pm2 = mws.summary.loc[stream_track_name, ['width_pm_phi1_cosphi2', 'width_pm_phi2']]

                # Mask for the stars with proper motions
                pm_mask = np.isfinite(stream_stars.pm_phi1_cosphi2) & np.isfinite(stream_stars.pm_phi2)

                # Extract proper motions of stream stars
                stars_pm1 = stream_stars[pm_mask].pm_phi1_cosphi2.to_value(units.mas / units.yr)
                stars_pm2 = stream_stars[pm_mask].pm_phi2.to_value(units.mas / units.yr)

                # Extract proper motions of stream track
                track_pm1 = stream.track.pm_phi1_cosphi2.to_value(units.mas / units.yr)
                track_pm2 = stream.track.pm_phi2.to_value(units.mas / units.yr)

                # Interpolate track proper motions vs phi1
                interp_pm1 = np.interp(stream_stars.phi1.deg[pm_mask], track_phi1, track_pm1)
                interp_pm2 = np.interp(stream_stars.phi1.deg[pm_mask], track_phi1, track_pm2)

                # Distance in pm directions
                pm1_separation = np.abs(stars_pm1 - interp_pm1)
                pm2_separation = np.abs(stars_pm2 - interp_pm2)

                # Update in_stream_mask to only include stars that are also consistent in proper motion with the stream track
                in_stream_mask[pm_mask] &= (pm1_separation < width_pm1 / 2) & (pm2_separation < width_pm2 / 2)
            except:
                pass
        # Get the indices of the potential stream stars in the original arrays
        indices = np.where(in_footprint_mask)[0][in_stream_mask]
        del in_footprint_mask, track_phi1, in_stream_mask, stream_stars  # Free memory
        gc.collect()  # Force garbage collection

        # Assign the stream IDs and probabilities to the members
        print(f'... processing stream {stream_id + 1}/{len(mws)} ({stream_track_name}) | assigning stream IDs and memberships to potential stream members                ', end='\r')
        unassigned_bool = True
        indices_stream_ids = galstreams_members_stream_ids_gdr3[indices]
        for j in range(galstreams_members_stream_ids_gdr3.shape[1]):
            # Find which members can be assigned to this column
            unassigned_indices_bool = indices_stream_ids[:, j] == max_galstream_stream_ID + 1

            # Assign the stream IDs and probabilities to the unassigned members for this column
            unassigned_indices = indices[unassigned_indices_bool]
            galstreams_members_stream_ids_gdr3[unassigned_indices, j] = stream_id
            galstreams_members_stream_membership_gdr3[unassigned_indices, j] = 1

            # Update the arrays to only include those that are still unassigned
            indices = indices[~unassigned_indices_bool]
            indices_stream_ids = indices_stream_ids[~unassigned_indices_bool]

            # If all members have been assigned, break
            if indices_stream_ids.shape[0] == 0:
                unassigned_bool = False
                break
        del indices_stream_ids  # Free memory
        gc.collect()  # Force garbage collection

        if unassigned_bool:
            # Add another column to the arrays
            galstreams_members_stream_ids_gdr3 = np.concatenate(
                (galstreams_members_stream_ids_gdr3, np.full((subsample_mask.size, 1), max_galstream_stream_ID + 1, dtype=np.int64)),
                axis=1
            )
            galstreams_members_stream_membership_gdr3 = np.concatenate(
                (galstreams_members_stream_membership_gdr3, np.zeros((subsample_mask.size, 1), dtype=np.bool_)),
                axis=1
            )

            # Assign the stream IDs and memberships to the members
            galstreams_members_stream_ids_gdr3[indices, -1] = stream_id
            galstreams_members_stream_membership_gdr3[indices, -1] = 1
    del ra, dec, mu_ra, mu_dec, cos_dec, mws, indices  # Free memory
    gc.collect()  # Force garbage collection
    
    # Get the stream IDs and membership for the subsample in this work
    galstreams_members_stream_ids_subsample = galstreams_members_stream_ids_gdr3[subsample_mask]
    galstreams_members_stream_membership_subsample = galstreams_members_stream_membership_gdr3[subsample_mask]
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total size for each galstreams stream
    print("... pre-computing the total size for each galstreams stream                                                                   ")
    galstreams_stream_sizes_full = np.bincount(
        galstreams_members_stream_ids_gdr3.ravel(),
        weights=galstreams_members_stream_membership_gdr3.ravel(),
        minlength=max_galstream_stream_ID + 1
    )[:max_galstream_stream_ID + 1]  # (N_streams,)
    del galstreams_members_stream_ids_gdr3, galstreams_members_stream_membership_gdr3  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the size for each galstreams stream in the overlap with the subsample
    print("... pre-computing the size for each galstreams stream in the subsample of this work")
    galstreams_stream_sizes_in_subsample = np.bincount(
        galstreams_members_stream_ids_subsample.ravel(),
        weights=galstreams_members_stream_membership_subsample.ravel(),
        minlength=max_galstream_stream_ID + 1
    )[:max_galstream_stream_ID + 1]  # (N_streams,)

    # Calculate the coverage of each galstreams stream by the subsample of this work
    galstreams_stream_coverage = np.full_like(galstreams_stream_sizes_full, 0.0, dtype=np.float64)
    valid = galstreams_stream_sizes_full > 0
    galstreams_stream_coverage[valid] = (
        galstreams_stream_sizes_in_subsample[valid] /
        galstreams_stream_sizes_full[valid]
    )
    del galstreams_stream_sizes_full  # Free memory
    gc.collect()  # Force garbage collection

    # Save the intermediary results
    print('... saving intermediary results.\n')
    np.save(file_path_galstreams_members_stream_ids_subsample, galstreams_members_stream_ids_subsample)
    np.save(file_path_galstreams_members_stream_membership_subsample, galstreams_members_stream_membership_subsample)
    np.save(file_path_galstreams_stream_sizes_in_subsample, galstreams_stream_sizes_in_subsample)
    np.save(file_path_galstreams_stream_coverage, galstreams_stream_coverage)
    del galstreams_members_stream_ids_subsample, galstreams_members_stream_membership_subsample, galstreams_stream_sizes_in_subsample, galstreams_stream_coverage  # Free memory
    gc.collect()  # Force garbage collection

def plot_galstreams_streams_on_sky(overwrite=False):
    """
    Plot the galstreams streams on the sky.
    """
    # Check if plot already exists
    file_path_streams_on_sky = os.path.join(OUTPUT_PATH, "galstreams_streams_on_sky.png")
    if os.path.exists(file_path_streams_on_sky) and not overwrite:
        print(f"Galstreams streams on sky plot already exists at:\n\t{file_path_streams_on_sky} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting galstreams streams on the sky...")

    # Load the galstreams clustering output
    print("... loading galstreams clustering output for plotting")
    galstreams_members_stream_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_members_stream_ids_subsample.npy"))  # (N_subsample, max_appearances)
    galstreams_members_stream_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_members_stream_membership_subsample.npy"))  # (N_subsample, max_appearances)
    
    # Plot the streams on the sky
    plot_catalogue_structure_on_sky(
        galstreams_members_stream_ids_subsample,
        galstreams_members_stream_membership_subsample,
        file_path_streams_on_sky
    )

def compare_to_galstreams(overwrite=False):
    """
    Compare the clustering output to the galstreams catalogue.
    """
    # Check if comparison results already exist
    file_path_catalogue_rpj_allS = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_catalogue_rpj_allS.npy")
    file_path_catalogue_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_catalogue_idx_optimalS.npy")
    file_path_astrolink_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_astrolink_rpj_optimalS.npy")
    file_path_astrolink_idx_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_astrolink_idx_optimalS.npy")
    file_path_catalogue_rpj_optimalS = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_catalogue_rpj_optimalS.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_catalogue_rpj_allS) and
                 os.path.exists(file_path_catalogue_idx_optimalS) and
                 os.path.exists(file_path_astrolink_rpj_optimalS) and
                 os.path.exists(file_path_astrolink_idx_optimalS) and
                 os.path.exists(file_path_catalogue_rpj_optimalS))
    if all_exist and not overwrite:
        print("Galstreams comparison results already exist at:")
        print(f"\t{file_path_catalogue_rpj_allS} ,")
        print(f"\t{file_path_catalogue_idx_optimalS} ,")
        print(f"\t{file_path_astrolink_rpj_optimalS} ,")
        print(f"\t{file_path_astrolink_idx_optimalS} , and")
        print(f"\t{file_path_catalogue_rpj_optimalS} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to the galstreams catalogue...")

    # Load the reduced galstreams data
    print("... loading reduced galstreams data")
    galstreams_members_stream_ids_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_members_stream_ids_subsample.npy"))  # (N_subsample, max_appearances)
    galstreams_members_stream_membership_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_members_stream_membership_subsample.npy"))  # (N_subsample, max_appearances)
    galstreams_stream_sizes_in_subsample = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_stream_sizes_in_subsample.npy")) # (N_streams,)

    # Choose comparison metric
    comparison_metric = 1  # 0 for recovery, 1 for purity (S_C=S_A), 2 for purity (S_C=all_catalogue_stars), 3 for Jaccard index (S_C=S_A), 4 for Jaccard index (S_C=all_catalogue_stars)
    
    # Make the comparison
    compare_to_catalogue_helper(
        galstreams_members_stream_ids_subsample,
        galstreams_members_stream_membership_subsample,
        galstreams_stream_sizes_in_subsample,
        comparison_metric,
        file_path_catalogue_rpj_allS,
        file_path_catalogue_idx_optimalS,
        file_path_astrolink_rpj_optimalS,
        file_path_astrolink_idx_optimalS,
        file_path_catalogue_rpj_optimalS
    )

def plot_galstreams_crossmatch_per_significance(overwrite=False):
    """
    Plot a summary the comparison between the AstroLink clustering output and the 
    galstreams catalogue as a function of the significance threshold S.
    """
    # Check if plot already exists
    file_path = os.path.join(OUTPUT_PATH, "galstreams_crossmatch_per_significance.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Galstreams comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Galstreams comparison results...")

    # Load comparison results
    print("... loading comparison results")
    RPJ = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_catalogue_rpj_allS.npy"))  # (N_sigmas, N_streams, 5)
    galstreams_stream_coverage = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_stream_coverage.npy"))

    # Load stream track names
    galstreams_stream_track_names = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_stream_track_names.npy"))  # (N_streams,)

    # Extract per-assumption and per-statistic arrays
    R  = RPJ[:, :, 0]
    P_full,  P_union = RPJ[:, :, 1], RPJ[:, :, 2]
    J_full,  J_union = RPJ[:, :, 3], RPJ[:, :, 4]

    # Evidence-weighted averages
    coverage_sum = np.sum(galstreams_stream_coverage)
    Rbar  = np.sum(R * galstreams_stream_coverage, axis=1) / coverage_sum
    Pbar_full  = np.sum(P_full * galstreams_stream_coverage, axis=1) / coverage_sum
    Pbar_union = np.sum(P_union * galstreams_stream_coverage, axis=1) / coverage_sum
    Jbar_full  = np.sum(J_full * galstreams_stream_coverage, axis=1) / coverage_sum
    Jbar_union = np.sum(J_union * galstreams_stream_coverage, axis=1) / coverage_sum

    print("... plotting combined statistics vs significance level")

    # Define streams and colors
    best_matching_streams = np.argsort(np.max(P_union, axis=0))[::-1][:4]
    stream_type_colors = {
        stream_name: f"C{index}"
        for index, stream_name in enumerate(galstreams_stream_track_names[best_matching_streams])
    }

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Jaccard
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_union,
            color='k', linestyle='solid', linewidth=1, zorder=3)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full,
            color='k', linestyle='solid', linewidth=1, zorder=3)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, Pbar_full, Pbar_union,
            color='k', alpha=0.3, zorder=3)

    # Cycle through specific streams and plot their Jaccard indices vs significance level
    for stream_name, color in stream_type_colors.items():
        # Find index of this stream
        stream_index = np.where(galstreams_stream_track_names == stream_name)[0][0]

        # Plot Jaccard indices for this stream
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, P_union[:, stream_index],
                color=color, linestyle='solid', linewidth=1, zorder=4)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, P_full[:, stream_index],
                color=color, linestyle='solid', linewidth=1, zorder=4)
        ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, P_full[:, stream_index], P_union[:, stream_index],
                color=color, alpha=0.3, zorder=4)

    # Matched fractions
    fraction_matched_union = np.mean(P_union > 0.5, axis=1)
    fraction_matched_full = np.mean(P_full > 0.5, axis=1)

    # Plot fraction of clusters of this type matched to above J=0.5
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_union,
            color='k', linestyle='dashed', linewidth=1, zorder=2)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full,
            color='k', linestyle='dashed', linewidth=1, zorder=2)
    ax.fill_between(SIGMA_THRESHOLDS_FOR_COMPARISONS, fraction_matched_full, fraction_matched_union,
            facecolor='none', hatch='//', edgecolor='k', linewidth=1, zorder=2)

    # Plot optimal S value on top
    ax.axvline(OPTIMAL_SIGMA_THRESHOLD, color='grey', linestyle='dotted', linewidth=0.75, zorder=4)
    ax.text(
        OPTIMAL_SIGMA_THRESHOLD, 1.01,
        f"S = {OPTIMAL_SIGMA_THRESHOLD}",
        color='grey', fontsize=10, ha='center', va='bottom', zorder=4
    )

    # Dummy handles for the legend
    handles = [
        Rectangle((0,0), 1, 1, facecolor=mcolors.to_rgba('k', alpha=0.3), edgecolor='k', linewidth=1, label='Purity'),
        Rectangle((0,0), 1, 1, facecolor='none', hatch='//', edgecolor='k', linewidth=1, label='Matched fraction'),
    ] + [plt.Line2D([], [], color='none', label=stream_name) for stream_name in stream_type_colors.keys()]

    # Create the legend
    leg = ax.legend(handles=handles, loc='upper right', frameon=False)

    # Recolor legend text entries for cluster types
    for text in leg.get_texts():
        label = text.get_text()
        if label in stream_type_colors:
            text.set_color(stream_type_colors[label])
            text.set_fontweight('bold')

    # Final formatting
    print("... saving figure.\n")
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Significance, $S$")
    ax.set_ylabel("Comparison statistic")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)
    gc.collect()


# === Summarise catalogue comparisons ===
def retrieve_relationship_classification(catalogue_idx, astrolink_idx):
    """
    Retrieve the relationship classification crossmatch results between one 
    catalogue and another.
    """
    # Best match mask
    has_match_mask = astrolink_idx != -1

    # Mask for one-to-one matches (AstroLink cluster is best matched to this catalogue cluster and vice versa)
    reciprocal_mask = np.zeros_like(has_match_mask, dtype=np.bool_)
    reciprocal_mask[has_match_mask] = (
        catalogue_idx[astrolink_idx[has_match_mask]] ==
        np.arange(astrolink_idx.size)[has_match_mask]
    )

    # Mask for fragmented matches (multiple "fragmented" AstroLink clusters are best matched to the same catalogue cluster)
    counts_C = np.bincount(
        catalogue_idx[catalogue_idx != -1],
        minlength=astrolink_idx.size
    )
    fragmented_mask = counts_C > 1
    fragmented_mask &= has_match_mask

    # Mask for merged matches (AstroLink is the best match of multiple "merged" catalogue clusters)
    counts_A = np.bincount(
        astrolink_idx[has_match_mask],
        minlength=catalogue_idx.size
    )
    merged_mask = np.zeros_like(has_match_mask, dtype=np.bool_)
    merged_mask[has_match_mask] = (
        counts_A[astrolink_idx[has_match_mask]] > 1
    )

    # Store the mask information in a single array for easy access in the table printing function
    relationship_classification = np.zeros(astrolink_idx.size, dtype=np.uint8)
    relationship_classification |= reciprocal_mask.astype(np.uint8) << 0
    relationship_classification |= fragmented_mask.astype(np.uint8) << 1
    relationship_classification |= merged_mask.astype(np.uint8) << 2
    relationship_classification |= has_match_mask.astype(np.uint8) << 3

    return relationship_classification

def construct_relationship_classifications(overwrite=False):
    """
    Placeholder
    """
    # Check if plot already exists
    file_path_Hunt2024_relationship_classification = os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_relationship_classification.npy")
    file_path_UCC_relationship_classification = os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_relationship_classification.npy")
    file_path_Vasiliev2021_relationship_classification = os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_relationship_classification.npy")
    file_path_Battaglia2021_relationship_classification = os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_relationship_classification.npy")
    file_path_galstreams_relationship_classification = os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_relationship_classification.npy")
    file_path_combined_astrolink_relationship_classification = os.path.join(OUTPUT_PATH, "combined_astrolink_relationship_classification.npy")
    file_path_combined_astrolink_rpj = os.path.join(OUTPUT_PATH, "combined_astrolink_rpj.npy")
    file_path_combined_astrolink_cluster_type = os.path.join(OUTPUT_PATH, "combined_astrolink_cluster_type.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_Hunt2024_relationship_classification) and
                 os.path.exists(file_path_UCC_relationship_classification) and
                 os.path.exists(file_path_Vasiliev2021_relationship_classification) and
                 os.path.exists(file_path_Battaglia2021_relationship_classification) and
                 os.path.exists(file_path_galstreams_relationship_classification) and
                 os.path.exists(file_path_combined_astrolink_relationship_classification) and
                 os.path.exists(file_path_combined_astrolink_rpj) and
                 os.path.exists(file_path_combined_astrolink_cluster_type))
    if all_exist and not overwrite:
        print("Relationship classifications already exist at:")
        print(f"\t{file_path_Hunt2024_relationship_classification} ,")
        print(f"\t{file_path_UCC_relationship_classification} ,")
        print(f"\t{file_path_Vasiliev2021_relationship_classification} ,")
        print(f"\t{file_path_Battaglia2021_relationship_classification} ,")
        print(f"\t{file_path_galstreams_relationship_classification} ,")
        print(f"\t{file_path_combined_astrolink_relationship_classification} ,")
        print(f"\t{file_path_combined_astrolink_rpj} , and")
        print(f"\t{file_path_combined_astrolink_cluster_type} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Reducing catalogue comparison results to structural relationship classifications...")

    # Dictionary of file paths for the relationship classification arrays
    file_paths_relationship_classification = {
        "Hunt2024": file_path_Hunt2024_relationship_classification,
        "UCC": file_path_UCC_relationship_classification,
        "Vasiliev2021": file_path_Vasiliev2021_relationship_classification,
        "Battaglia2021": file_path_Battaglia2021_relationship_classification,
        "galstreams": file_path_galstreams_relationship_classification,
    }

    # Dictionary of file paths for the structure name arrays
    structure_name_files = {
        "Hunt2024": os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_clusters_names.npy"),
        "UCC": os.path.join(INTERMEDIATE_FILES_PATH, "UCC/UCC_clusters_names.npy"),
        "Vasiliev2021": os.path.join(INTERMEDIATE_FILES_PATH, "Vasiliev2021/Vasiliev2021_clusters_names.npy"),
        "Battaglia2021": os.path.join(INTERMEDIATE_FILES_PATH, "Battaglia2021/Battaglia2021_dwarfgalaxies_names.npy"),
        "galstreams": os.path.join(INTERMEDIATE_FILES_PATH, "galstreams/galstreams_stream_track_names.npy"),
    }

    # Translate structure type labels for Hunt2024 clusters from single-letter codes to descriptive labels
    Hunt2024_clusters_types = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "Hunt2024/Hunt2024_clusters_types.npy"), allow_pickle=True)  # (N_Hunt2024_clusters,)

    # Dictionary of structure type labels for each catalogue
    structure_type_labels = {
        "Hunt2024": Hunt2024_clusters_types,
        "UCC": 'o',
        "Vasiliev2021": 'g',
        "Battaglia2021": 'D',
        "galstreams": 's',
    }

    combined_astrolink_relationship_classification = np.zeros(0, dtype=np.uint8)  # Placeholder for combined relationship classification array for all catalogues (N_AstroLink_clusters,)

    # Loop over all catalogues
    for i, catalogue_name in enumerate(['Hunt2024', 'UCC', 'Vasiliev2021', 'Battaglia2021', 'galstreams']):
        # Load comparison arrays at optimal significance threshold
        print(f"... loading arrays for catalogue: {catalogue_name}")
        catalogue_idx = np.load(os.path.join(INTERMEDIATE_FILES_PATH, f"{catalogue_name}/{catalogue_name}_catalogue_idx_optimalS.npy"))  # (N_AstroLink_clusters,)
        astrolink_idx = np.load(os.path.join(INTERMEDIATE_FILES_PATH, f"{catalogue_name}/{catalogue_name}_astrolink_idx_optimalS.npy"))  # (N_catalogue_clusters,)

        # Retrieve the relationship classification for this catalogue
        print(f"... retrieving relationship classification for catalogue: {catalogue_name}")
        catalogue_relationship_classification = retrieve_relationship_classification(catalogue_idx, astrolink_idx) # (N_catalogue_clusters,)

        # Save the relationship classification for this catalogue
        print(f"... saving relationship classification for catalogue: {catalogue_name}")
        np.save(file_paths_relationship_classification[catalogue_name], catalogue_relationship_classification)

        # Update the combined astrolink cluster classification arrays
        print(f"... updating combined AstroLink cluster classification arrays with catalogue: {catalogue_name}")
        astrolink_relationship_classification = retrieve_relationship_classification(astrolink_idx, catalogue_idx) # (N_astrolink_clusters,)
        astrolink_rpj = np.load(os.path.join(INTERMEDIATE_FILES_PATH, f"{catalogue_name}/{catalogue_name}_astrolink_rpj_optimalS.npy")) # (N_astrolink_clusters, 5)
        astrolink_cluster_type = np.full(catalogue_idx.shape[0], "U", dtype=object)  # Default to "U' for "Unknown" for clusters not matched to any catalogue cluster
        matched = catalogue_idx != -1
        if i == 0:
            # Relationship classification
            combined_astrolink_relationship_classification = astrolink_relationship_classification

            # Comparison statistics
            combined_astrolink_rpj = astrolink_rpj[:, None, :]  # Add new axis for catalogues

            # Structure types
            astrolink_cluster_type[matched] = structure_type_labels[catalogue_name][catalogue_idx[matched]]
            combined_astrolink_cluster_type = astrolink_cluster_type
        else:
            # Relationship classification
            combined_astrolink_relationship_classification = np.column_stack(
                (combined_astrolink_relationship_classification, astrolink_relationship_classification)
            )

            # Comparison statistics
            combined_astrolink_rpj = np.concatenate(
                (combined_astrolink_rpj, astrolink_rpj[:, None, :]),
                axis=1
            )

            # Structure types
            astrolink_cluster_type[matched] = structure_type_labels[catalogue_name]
            combined_astrolink_cluster_type = np.column_stack(
                (combined_astrolink_cluster_type, astrolink_cluster_type)
            )
        
        # Save combined AstroLink cluster classification arrays after each catalogue is added
        print(f"... saving combined AstroLink cluster classification arrays.\n")
        np.save(file_path_combined_astrolink_relationship_classification, combined_astrolink_relationship_classification)
        np.save(file_path_combined_astrolink_rpj, combined_astrolink_rpj)
        np.save(file_path_combined_astrolink_cluster_type, combined_astrolink_cluster_type)

def plot_astrolink_clusters_by_structure_type_on_sky(overwrite=False):
    """
    Plot the AstroLink clusters on the sky, coloured by their structure type as 
    classified by their relationship to the comparison catalogues at the optimal 
    significance threshold.

    All structure types except "U" (Unknown) are plotted in a single 3x2 panel
    figure (with "d" and "r" combined). The "Unknown" class is plotted in a
    separate figure. Each figure contains exactly one legend showing the meaning
    of the colours in terms of the relationship classification system.
    """
    # Define dictionary to map from encoded structure type label to structure name
    structure_type_name_map = {
        "o": "Open clusters",
        "m": "Moving groups",
        "g": "Globular clusters",
        "d": "Too distant to classify",
        "r": "Rejected",
        "D": "Dwarf galaxies",
        "s": "Stellar streams",
        "U": "Unknown"
    }

    # Define dictionary to map from relationship classification to plotting colour
    relationship_colour_map = {
        "True 1-1":    mcolors.to_rgb('C2'),
        "Frag 1-1":    mcolors.to_rgb('C0'),
        "Merg 1-1":    mcolors.to_rgb('C9'),
        "F & M 1-1":   mcolors.to_rgb('C4'),
        "True Frag":   mcolors.to_rgb('C6'),
        "True Merg":   mcolors.to_rgb('C1'),
        "Frag & Merg": mcolors.to_rgb('C8'),
        "Isolated":    mcolors.to_rgb('C3')
    }

    # Check if plot already exists
    multi_panel_file_path = os.path.join(OUTPUT_PATH, "AstroLink_all_structure_types_except_unknown_matches_on_sky.png")
    unknown_file_path = os.path.join(OUTPUT_PATH,"AstroLink_unknown_matches_on_sky.png")

    # Skip processing if output files already exist
    if (os.path.exists(multi_panel_file_path) and
        os.path.exists(unknown_file_path) and not overwrite):
        print("AstroLink clusters by structure type on sky plots already exist at:")
        print(f"\t{multi_panel_file_path} , and")
        print(f"\t{unknown_file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return

    print("Plotting AstroLink clusters by structure type on the sky...")

    # Load combined relationship classification and cluster type arrays
    print("... loading combined relationship classification and cluster type arrays")
    combined_astrolink_relationship_classification = np.load(os.path.join(OUTPUT_PATH, "combined_astrolink_relationship_classification.npy"))  # (N_AstroLink_clusters, N_catalogues)
    combined_astrolink_rpj = np.load(os.path.join(OUTPUT_PATH, "combined_astrolink_rpj.npy"))  # (N_AstroLink_clusters, N_catalogues, 5)
    combined_astrolink_cluster_type = np.load(os.path.join(OUTPUT_PATH, "combined_astrolink_cluster_type.npy"), allow_pickle=True)  # (N_AstroLink_clusters, N_catalogues)

    # Reduce the combined AstroLink cluster classification arrays to the best-matching catalogue/structure
    print("... reducing combined AstroLink cluster classification arrays to best-matching catalogue/structure")
    best_catalogue = combined_astrolink_rpj[:, :, 3].argmax(axis=1)
    rows = np.arange(combined_astrolink_rpj.shape[0])
    best_astrolink_relationship_classification = combined_astrolink_relationship_classification[rows, best_catalogue]
    best_astrolink_rpj = combined_astrolink_rpj[rows, best_catalogue]
    best_astrolink_cluster_type = combined_astrolink_cluster_type[rows, best_catalogue]

    # Extract relationship classifications
    print("... extracting relationship classifications for best-matching catalogue/structure")
    R = (best_astrolink_relationship_classification & 1) != 0
    F = (best_astrolink_relationship_classification & 2) != 0
    M = (best_astrolink_relationship_classification & 4) != 0
    H = (best_astrolink_relationship_classification & 8) != 0

    true_reciprocal  = R & ~F & ~M
    frag_reciprocal  = R & F & ~M
    merg_reciprocal  = R & M & ~F
    f_n_M_reciprocal = R & F & M
    true_fragmented  = F & ~R & ~M
    true_merged      = M & ~R & ~F
    frag_and_merged  = F & M & ~R
    isolated_match   = H & ~R & ~F & ~M
    no_best_match    = ~H

    # Define marker face colours
    print("... defining marker colours based on relationship classification")
    cluster_facecolours = np.ones((best_astrolink_cluster_type.shape[0], 3), dtype=np.float64)

    cluster_facecolours[true_reciprocal]  = relationship_colour_map["True 1-1"]
    cluster_facecolours[frag_reciprocal]  = relationship_colour_map["Frag 1-1"]
    cluster_facecolours[merg_reciprocal]  = relationship_colour_map["Merg 1-1"]
    cluster_facecolours[f_n_M_reciprocal] = relationship_colour_map["F & M 1-1"]
    cluster_facecolours[true_fragmented]  = relationship_colour_map["True Frag"]
    cluster_facecolours[true_merged]      = relationship_colour_map["True Merg"]
    cluster_facecolours[frag_and_merged]  = relationship_colour_map["Frag & Merg"]
    cluster_facecolours[isolated_match]   = relationship_colour_map["Isolated"]

    cluster_edgecolours = np.full(cluster_facecolours.shape, (0, 0, 0))

    # Whiten markers for clusters with J < 0.5
    whiten_factor = 0.75
    potential_mismatch = best_astrolink_rpj[:, 3] < 0.5
    cluster_facecolours[potential_mismatch] *= 1 - whiten_factor
    cluster_facecolours[potential_mismatch] += whiten_factor
    cluster_edgecolours[potential_mismatch] = whiten_factor

    no_best_match_indices = np.where(no_best_match)[0]
    for i, idx in enumerate(no_best_match_indices):
        cluster_facecolours[idx] = mcolors.to_rgb(f"C{i}")

    # Load AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = loadAstroLinkObject(os.path.join(OUTPUT_PATH, "astrolink_object.npz"))

    # Load coordinates
    print("... loading required arrays for plotting")
    subsample_mask = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "subsample_mask.npy"))
    galactic_coordinates = np.load(os.path.join(INTERMEDIATE_FILES_PATH, "gdr3_galactic_coordinates.npy"))[subsample_mask]
    del subsample_mask
    gc.collect()

    print("... converting galactic coordinates to radians for Mollweide projection")
    galactic_coordinates = np.deg2rad(galactic_coordinates)
    longitude_wrap_bool = galactic_coordinates[:, 0] > np.pi
    galactic_coordinates[longitude_wrap_bool, 0] -= 2*np.pi
    galactic_coordinates[:, 0] *= -1
    del longitude_wrap_bool
    gc.collect()

    # ================================
    # Create 3x2 multi-panel figure
    # ================================
    print("... creating 3x2 multi-panel figure for all structure types except 'Unknown'")

    multi_panel_structure_groups = [
        ("o",),
        ("g",),
        ("m",),
        ("D",),
        ("d", "r"),
        ("s",)
    ]

    fig, axes = plt.subplots(
        3, 2,
        figsize=(12, 10),
        subplot_kw={'projection': 'mollweide'}
    )

    axes = axes.flatten()

    for ax, structure_group in zip(axes, multi_panel_structure_groups):

        structure_mask = np.isin(best_astrolink_cluster_type, structure_group)

        for i in np.where(structure_mask)[0]:
            if i > 0:
                clst = clusterer.clusters[i]
                clusterMembers = clusterer.ordering[clst[0]:clst[1]]

                ax.scatter(
                    *galactic_coordinates[clusterMembers].T,
                    facecolor=cluster_facecolours[i],
                    edgecolor=cluster_edgecolours[i],
                    s=0.75, lw=0.075
                )

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        if len(structure_group) == 1:
            ax.set_title(structure_type_name_map[structure_group[0]])
        else:
            ax.set_title("Too distant / Rejected")

    # Create scatter-style legend handles (matching plotted markers)
    upper_legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=colour,
            markeredgecolor='black',
            markersize=8,
            label=label
        )
        for label, colour in list(relationship_colour_map.items())[:4]
    ]

    lower_legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=colour,
            markeredgecolor='black',
            markersize=8,
            label=label
        )
        for label, colour in list(relationship_colour_map.items())[4:]
    ]

    # First legend (top four panels intersection)
    legend1 = fig.legend(
        handles=upper_legend_handles,
        loc='center',
        bbox_to_anchor=(0.5, 0.65675),   # adjust slightly if needed
        ncol=2,
        frameon=False
    )

    # Second legend (lower four panels intersection)
    legend2 = fig.legend(
        handles=lower_legend_handles,
        loc='center',
        bbox_to_anchor=(0.5, 0.33),   # adjust slightly if needed
        ncol=2,
        frameon=False
    )

    # Ensure both legends are kept
    fig.add_artist(legend1)
    fig.add_artist(legend2)

    plt.tight_layout()
    plt.savefig(multi_panel_file_path, dpi=500)
    plt.close()
    gc.collect()

    print(f"... saved multi-panel plot to {multi_panel_file_path}")

    # ================================
    # Create separate Unknown figure
    # ================================
    print("... creating separate 'Unknown' structure type figure")

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={'projection': 'mollweide'}
    )

    structure_mask = best_astrolink_cluster_type == "U"

    for i in np.where(structure_mask)[0]:
        if i > 0:
            clst = clusterer.clusters[i]
            clusterMembers = clusterer.ordering[clst[0]:clst[1]]

            ax.scatter(
                *galactic_coordinates[clusterMembers].T,
                facecolor=cluster_facecolours[i],
                edgecolor=cluster_edgecolours[i],
                s=0.75, lw=0.075
            )

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Unknown")

    plt.tight_layout()
    plt.savefig(unknown_file_path, dpi=500)
    plt.close()
    gc.collect()

    print(f"... saved 'Unknown' plot to {unknown_file_path}.\n")

def print_relationship_classification_table():
    """
    Print a summary table of the relationship classifications between the 
    AstroLink clustering output and each of the comparison catalogues at the 
    optimal significance threshold.
    """
    # Print description
    print("Relationship classification summary table at optimal significance threshold...")
    print(" " * 36 + "-" * 22)
    print(f"                                   | {'N(J>0.5, S_C=All)':>20} |")
    print(f"... format of each category entry: | {'N':<20} | ")
    print(f"                                   | {'N(J>0.5, S_C=S_A)':>20} |")
    print(" " * 36 + "-" * 22)
    
    # Print divider
    print("... " + "-" * 157)

    # Print formatted column headings
    print(
        "    | "
        f"{'Map direction':<13} | "
        f"{'All':>11} | "
        f"{'True 1-1':>11} | "
        f"{'Frag 1-1':>11} | "
        f"{'Merg 1-1':>11} | "
        f"{'F & M 1-1':>11} | "
        f"{'True Frag':>11} | "
        f"{'True Merg':>11} | "
        f"{'Frag & Merg':>11} | "
        f"{'Isolated':>11} | "
        f"{'No match':>11} |"
    )

    # Print divider
    print("    " + "-" * 157)

    # Loop over all catalogues
    for catalogue_name in ['Hunt2024', 'UCC', 'Vasiliev2021', 'Battaglia2021', 'galstreams']:
        # Load relationship classification and catalogue RPJ at optimal significance threshold
        catalogue_relationship_classification = np.load(os.path.join(INTERMEDIATE_FILES_PATH, f"{catalogue_name}/{catalogue_name}_relationship_classification.npy"))  # (N_catalogue_clusters,)
        catalogue_rpj = np.load(os.path.join(INTERMEDIATE_FILES_PATH, f"{catalogue_name}/{catalogue_name}_catalogue_rpj_optimalS.npy"))  # (N_catalogue_clusters, 5)

        # Extract each relationship classification
        R = (catalogue_relationship_classification & 1) != 0
        F = (catalogue_relationship_classification & 2) != 0
        M = (catalogue_relationship_classification & 4) != 0
        H = (catalogue_relationship_classification & 8) != 0

        true_reciprocal  = R & ~F & ~M
        frag_reciprocal  = R & F & ~M
        merg_reciprocal  = R & M & ~F
        f_n_M_reciprocal = R & F & M
        true_fragmented  = F & ~R & ~M
        true_merged      = M & ~R & ~F
        frag_and_merged  = F & M & ~R
        isolated_match   = H & ~R & ~F & ~M
        no_best_match    = ~H

        # Print formatted top row
        print(
            "    | "
            f"{catalogue_name:<13} | "
            f"{catalogue_relationship_classification.size:>11} | "
            f"{true_reciprocal.sum():>11} | "
            f"{frag_reciprocal.sum():>11} | "
            f"{merg_reciprocal.sum():>11} | "
            f"{f_n_M_reciprocal.sum():>11} | "
            f"{true_fragmented.sum():>11} | "
            f"{true_merged.sum():>11} | "
            f"{frag_and_merged.sum():>11} | "
            f"{isolated_match.sum():>11} | "
            f"{no_best_match.sum():>11} |"
        )

        # Print formatted middle row
        good_match_upper = catalogue_rpj[:, 4] > 0.5
        print(
            "    | "
            f"{'->':<13} | "
            f"{good_match_upper.sum():>11} | "
            f"{good_match_upper[true_reciprocal].sum():>11} | "
            f"{good_match_upper[frag_reciprocal].sum():>11} | "
            f"{good_match_upper[merg_reciprocal].sum():>11} | "
            f"{good_match_upper[f_n_M_reciprocal].sum():>11} | "
            f"{good_match_upper[true_fragmented].sum():>11} | "
            f"{good_match_upper[true_merged].sum():>11} | "
            f"{good_match_upper[frag_and_merged].sum():>11} | "
            f"{good_match_upper[isolated_match].sum():>11} | "
            f"{good_match_upper[no_best_match].sum():>11} |"
        )

        # Print formatted lower row
        good_match_lower = catalogue_rpj[:, 3] > 0.5
        print(
            "    | "
            f"{'Ours':<13} | "
            f"{good_match_lower.sum():>11} | "
            f"{good_match_lower[true_reciprocal].sum():>11} | "
            f"{good_match_lower[frag_reciprocal].sum():>11} | "
            f"{good_match_lower[merg_reciprocal].sum():>11} | "
            f"{good_match_lower[f_n_M_reciprocal].sum():>11} | "
            f"{good_match_lower[true_fragmented].sum():>11} | "
            f"{good_match_lower[true_merged].sum():>11} | "
            f"{good_match_lower[frag_and_merged].sum():>11} | "
            f"{good_match_lower[isolated_match].sum():>11} | "
            f"{good_match_lower[no_best_match].sum():>11} |"
        )

        # Print divider
        print("    |" + "-" * 155 + "|")
    
    # Load combined AstroLink relationship classification and combined AstroLink RPJ at optimal significance threshold
    combined_astrolink_relationship_classification = np.load(os.path.join(OUTPUT_PATH, "combined_astrolink_relationship_classification.npy"))  # (N_astrolink_clusters, N_catalogues)
    combined_astrolink_rpj = np.load(os.path.join(OUTPUT_PATH, "combined_astrolink_rpj.npy"))  # (N_astrolink_clusters, N_catalogues, 5)

    # Reduce the combined AstroLink cluster classification arrays to the best-matching catalogue/structure for each AstroLink cluster
    best_catalogue = combined_astrolink_rpj[:, :, 3].argmax(axis=1)  # (N_AstroLink_clusters,)
    rows = np.arange(combined_astrolink_rpj.shape[0])
    best_astrolink_relationship_classification = combined_astrolink_relationship_classification[rows, best_catalogue] # (N_AstroLink_clusters,)
    best_astrolink_rpj = combined_astrolink_rpj[rows, best_catalogue] # (N_AstroLink_clusters, 5)

    # Extract each relationship classification
    R = (best_astrolink_relationship_classification & 1) != 0
    F = (best_astrolink_relationship_classification & 2) != 0
    M = (best_astrolink_relationship_classification & 4) != 0
    H = (best_astrolink_relationship_classification & 8) != 0

    true_reciprocal  = R & ~F & ~M
    frag_reciprocal  = R & F & ~M
    merg_reciprocal  = R & M & ~F
    f_n_M_reciprocal = R & F & M
    true_fragmented  = F & ~R & ~M
    true_merged      = M & ~R & ~F
    frag_and_merged  = F & M & ~R
    isolated_match   = H & ~R & ~F & ~M
    no_best_match    = ~H

    # Print formatted top row
    print(
        "    | "
        f"{'Ours':<13} | "
        f"{best_astrolink_relationship_classification.size - 1:>11} | "  # Subtract 1 to exclude the "cluster" containing all stars
        f"{true_reciprocal.sum():>11} | "
        f"{frag_reciprocal.sum():>11} | "
        f"{merg_reciprocal.sum():>11} | "
        f"{f_n_M_reciprocal.sum():>11} | "
        f"{true_fragmented.sum():>11} | "
        f"{true_merged.sum():>11} | "
        f"{frag_and_merged.sum():>11} | "
        f"{isolated_match.sum():>11} | "
        f"{no_best_match.sum() - 1:>11} |"  # Subtract 1 to exclude the "cluster" containing all stars
    )

    # Print formatted middle row
    good_match_upper = best_astrolink_rpj[:, 4] > 0.5
    print(
        "    | "
        f"{'->':<13} | "
        f"{good_match_upper.sum():>11} | "
        f"{good_match_upper[true_reciprocal].sum():>11} | "
        f"{good_match_upper[frag_reciprocal].sum():>11} | "
        f"{good_match_upper[merg_reciprocal].sum():>11} | "
        f"{good_match_upper[f_n_M_reciprocal].sum():>11} | "
        f"{good_match_upper[true_fragmented].sum():>11} | "
        f"{good_match_upper[true_merged].sum():>11} | "
        f"{good_match_upper[frag_and_merged].sum():>11} | "
        f"{good_match_upper[isolated_match].sum():>11} | "
        f"{good_match_upper[no_best_match].sum():>11} |"
    )

    # Print formatted lower row
    good_match_lower = best_astrolink_rpj[:, 3] > 0.5
    print(
        "    | "
        f"{'All others':<13} | "
        f"{good_match_lower.sum():>11} | "
        f"{good_match_lower[true_reciprocal].sum():>11} | "
        f"{good_match_lower[frag_reciprocal].sum():>11} | "
        f"{good_match_lower[merg_reciprocal].sum():>11} | "
        f"{good_match_lower[f_n_M_reciprocal].sum():>11} | "
        f"{good_match_lower[true_fragmented].sum():>11} | "
        f"{good_match_lower[true_merged].sum():>11} | "
        f"{good_match_lower[frag_and_merged].sum():>11} | "
        f"{good_match_lower[isolated_match].sum():>11} | "
        f"{good_match_lower[no_best_match].sum():>11} |"
    )
    
    # Print divider
    print("    " + "-" * 157)


# === Run script ===
if __name__ == "__main__":
    # Ensure paths exist
    os.makedirs(INTERMEDIATE_FILES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

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
    calculate_total_selection_function()
    plot_total_selection_function()

    # Construct input data to be passed to AstroLink
    calculate_contracted_subspaces_and_errors(True)
    construct_data_space(True)

    # Apply AstroLink to subsample and plot of cluster properties
    apply_astrolink_to_data(True)
    plot_astrolink_prominence_model_fit(True)
    plot_astrolink_cluster_labels_on_sky(True)
    plot_astrolink_cluster_proper_motions_on_sky(True)
    plot_astrolink_cluster_radial_velocities_on_sky(True)

    # Compare to Hunt & Reffert (2024)
    prepare_Hunt2024_for_comparison()
    plot_Hunt2024_clusters_on_sky()
    compare_to_Hunt2024(True)
    plot_Hunt2024_crossmatch_per_significance(True)

    # Compare to Unified Cluster Catalogue
    prepare_UCC_for_comparison()
    plot_UCC_clusters_on_sky()
    compare_to_UCC(True)
    plot_UCC_crossmatch_per_significance(True)

    # Compare to Vasiliev & Baumgardt (2021)
    prepare_Vasiliev2021_for_comparison()
    plot_Vasiliev2021_clusters_on_sky()
    compare_to_Vasiliev2021(True)
    plot_Vasiliev2021_crossmatch_per_significance(True)

    # Compare to Battaglia et al. (2021)
    prepare_Battaglia2021_for_comparison()
    plot_Battaglia2021_dwarfgalaxies_on_sky()
    compare_to_Battaglia2021(True)
    plot_Battaglia2021_crossmatch_per_significance(True)

    # Compare to galstreams catalogue
    prepare_galstreams_for_comparison()
    plot_galstreams_streams_on_sky()
    compare_to_galstreams(True)
    plot_galstreams_crossmatch_per_significance(True)

    # Summarise catalogue comparisons
    construct_relationship_classifications(True)
    plot_astrolink_clusters_by_structure_type_on_sky(True)
    print_relationship_classification_table()