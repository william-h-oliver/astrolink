# Standard imports
import os
import sys

# Restarts the script with a fresh interpreter state and forces the number of threads to be used.
# (this shouldn't actually be necessary, but is included for full control in case of a misbehaving environment)
PARALLEL_WORKERS = min(os.cpu_count(), 48)  # Use up to 48 workers or all available CPUs, whichever is smaller
if "THREAD_CONTROL_INIT" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = f"{PARALLEL_WORKERS}"
    os.environ["NUMBA_NUM_THREADS"] = f"{PARALLEL_WORKERS}"
    os.environ["NUMBA_DEFAULT_NUM_THREADS"] = f"{PARALLEL_WORKERS}"
    os.environ["THREAD_CONTROL_INIT"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from numba import njit, set_num_threads
set_num_threads(PARALLEL_WORKERS) # For some reason this is necessary to get pykdtree to use the correct number of threads

# Remaining standard imports
import gc
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import re
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm, beta
from scipy.special import gamma, digamma
from pykdtree.kdtree import KDTree
from sklearn import get_config
from sklearn.utils import gen_batches

# Astro-specific imports
from astropy.table import Table # Works using v6.1.3, but v7.1.0 seems to try and convert 'null' values to float before using fill_values
from astropy.coordinates import SkyCoord
import astropy.units as u
from gaiaunlimited.selectionfunctions import m10_to_completeness

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

# AstroLink imports
from astrolink import AstroLink
from astrolink import io
from astrolink import visualize


# === Define script configuration ===
# User-defined paths
GDR3_CATALOGUE_PATH = "/home/_data/Gaia/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"  # Path to raw gdr3 catalogue files
AUXILLARY_CATALOGUES_PATH = "/home/williamoliver_data/gaia_clustering/auxillary_catalogues/"  # Path to auxillary catalogues (e.g. Bailer-Jones GEDR3 distances, Hunt+2024 open clusters)
OUTPUT_PATH = "/home/williamoliver_data/gaia_clustering/"  # Path to output files

# Auto-defined paths
REDUCED_CATALOGUE_PATH = os.path.join(OUTPUT_PATH, "catalogue_files/")  # Path to reduced catalogue numpy files
SUBSAMPLE_PATH = os.path.join(OUTPUT_PATH, "subsample_files/")  # Path to numpy files of subsample from full catalogue
CLUSTERING_PATH = os.path.join(OUTPUT_PATH, "clustering_files/")  # Path to AstroLink output files
FIGURES_PATH = os.path.join(OUTPUT_PATH, "figures/")  # Path to figures

# Working memory for k-nearest-neighbour retrieval
WORKING_MEMORY = get_config()["working_memory"]  # Default is 1GB, but can be set to a higher value in sklearn config

# Pipeline constants
KNN_FOR_SELECTION_FUNCTION = 32 # Number of nearest neighbors for selection function calculations
SURVEY_SF_LOWER_LIMIT = 0.99 # Empirical survey selection function lower limit for subsample stars
SUBSAMPLE_RUWE_THRESHOLD = 1.2 # RUWE threshold for subsample stars
HEALPIX_LEVEL = 12 # HEALPix level for on-sky plotting
KNN_FOR_ASTROLINK = 10 # Number of nearest neighbors for AstroLink
SIGMA_FOR_ASTROLINK = 4 # Significance level for AstroLink
SIGMA_THRESHOLDS_FOR_COMPARISONS = np.linspace(2, 10, 81)  # Significance levels from 2 to 10 to be used when comparing to existing cluster catalogues



# === Reduce GDR3 and Bailer-Jones GEDR3 catalogues to numpy files ===
def reduce_gdr3_catalogue_to_numpy_files(overwrite=False):
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
        'photometry': ['phot_g_mean_mag'],
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
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = [
            executor.submit(_process_single_file, file_path, column_groups)
            for file_path in file_paths
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

def _process_single_file(file_path, column_groups):
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

    print(f"... [PROCESS] {chunk_name} — starting in PID {os.getpid():<15}", end='\r')
    
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

def reduce_bailerjones_gedr3_distances_to_numpy_files(overwrite=False):
    """
    Reads the Bailer-Jones et al. 2021 GEDR3 distances dump file, converts 
    columns to numpy arrays. Then re-index Bailer-Jones arrays to match 
    GDR3 source IDs.
    """
    # Define column groups for reduction
    column_groups = {
        'source_ids': ['source_id'],
        'r_med_geo': ['r_med_geo'],
        'r_lo_high_geo': ['r_lo_geo', 'r_hi_geo'],
        #'r_med_photogeo': ['r_med_photogeo'],
        #'r_lo_high_photogeo': ['r_lo_photogeo', 'r_hi_photogeo']
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
    bailerjones_gedr3_distances_file = os.path.join(AUXILLARY_CATALOGUES_PATH, 'gedr3dist.dump.gz')
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
def calculate_empirical_survey_selection_function(overwrite=True):
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
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))           # shape (n,)
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
    chunk_n_rows = min(int(WORKING_MEMORY * (2**20) // (16 * KNN_FOR_SELECTION_FUNCTION)), n)
    batches = list(gen_batches(n, chunk_n_rows))
    num_batches = len(batches)

    # Compute m10 for each star as median G of neighbors with <11 transits
    for i, sl in enumerate(batches):
        print(f"... computing m10 values for each star -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        sqr_dists, idx = tree.query(xyz_stars[sl], k=KNN_FOR_SELECTION_FUNCTION, sqr_dists=True)
        del sqr_dists  # Free memory
        gc.collect()  # Force garbage collection

        # Median G-band magnitude of neighbors
        m10_stars[sl] = np.median(G_band_magnitudes[valid_for_kNN[idx]], axis=1)
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
    chunk_n_rows = min(int(WORKING_MEMORY * (2**20) // (16 * KNN_FOR_SELECTION_FUNCTION)), npix)
    batches = list(gen_batches(npix, chunk_n_rows))
    num_batches = len(batches)

    # Initialize m10 array for HEALPix pixels
    print("... initializing m10 array for HEALPix pixels")
    m10_healpix = np.empty(npix)

    # Compute m10 for each HEALPix pixel as median G of neighbors with <11 transits
    for i, sl in enumerate(batches):
        print(f"... computing m10 values for HEALPix pixels -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        _, idx = tree.query(xyz_healpix[sl], k=KNN_FOR_SELECTION_FUNCTION, sqr_dists=True)

        # Median G-band magnitude of neighbors
        m10_healpix[sl] = np.median(G_band_magnitudes[valid_for_kNN[idx]], axis=1)

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
def construct_subsample_from_full_catalogue(overwrite=True):
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
    selection_function = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "empirical_survey_selection_function.npy"))  # (n,)
    galactic_coords = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_galactic_coordinates.npy"))  # (n, 2) in degrees
    r_med_geo = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_med_geo.npy"))  # (n,)
    proper_motions = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_proper_motions.npy"))  # (n, 2) in mas/yr
    ruwe = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_ruwe.npy"))  # (n,)
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy")) # (n,)

    # Create boolean mask for S_Gaia > threshold, ruwe < threshold, and valid astrometric data
    subsample_mask = selection_function > SURVEY_SF_LOWER_LIMIT
    subsample_mask &= ruwe < SUBSAMPLE_RUWE_THRESHOLD
    subsample_mask &= np.isfinite(r_med_geo)
    subsample_mask &= np.isfinite(proper_motions).all(axis=1)
    subsample_mask &= np.isfinite(G_band_magnitudes)

    # Save mask
    np.save(mask_path, subsample_mask)
    print(f"... saved subsample mask to {mask_path} (selected {subsample_mask.sum()} stars).\n")

def calculate_subsample_selection_function(overwrite=True):
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
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))  # shape (n,)
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
    x_scale = 4*np.pi/npix # Scale factor for Cartesian coordinates
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
    chunk_n_rows = min(int(WORKING_MEMORY * (2**20) // (16 * KNN_FOR_SELECTION_FUNCTION)), n)
    batches = list(gen_batches(n, chunk_n_rows))
    num_batches = len(batches)

    # Compute subsample selection function for each star as fraction of neighbourhood in subsample
    for i, sl in enumerate(batches):
        print(f"... computing subsample selection function for each star -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        _, idx = tree.query(comp_stars[sl], k=KNN_FOR_SELECTION_FUNCTION, sqr_dists=True)

        # Fraction of neighbours in subsample
        subsample_sf[valid_gmag[sl]] = subsample_mask[valid_gmag[idx]].sum(axis=1) / KNN_FOR_SELECTION_FUNCTION

        # Delete temporary variables to free memory
        del _, idx
        gc.collect()
    print(f"... subsample selection function range: {np.nanmin(subsample_sf):.3f} -- {np.nanmax(subsample_sf):.3f}")

    # Save subsample selection function for stars
    print(f"... saving subsample selection function for stars to {file_subsample_sf_stars} (shape: {subsample_sf.shape}).\n")
    np.save(file_subsample_sf_stars, subsample_sf)
    del subsample_sf, comp_stars, tree  # Free memory
    gc.collect()  # Force garbage collection


# === Calculate total selection function for subsample ===
def calculate_total_selection_function_for_subsample(overwrite=True):
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
    if os.path.exists(file_nsub) and os.path.exists(file_nmw) and os.path.exists(file_total_sf_mean) and os.path.exists(file_total_sf_var) and os.path.exists(file_total_sf_mean_healpix) and os.path.exists(file_total_sf_var_healpix) and not overwrite:
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
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))
    survey_sf = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "empirical_survey_selection_function.npy"))
    subsample_sf = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_selection_function.npy"))

    # Identify stars with valid G magnitude
    print("... identifying valid G-band magnitudes")
    valid_gmag = np.isfinite(G_band_magnitudes)
    del G_band_magnitudes  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate the inverse of the empirical survey selection function for the subsample
    inverse_survey_sf = 1 / np.sqrt(survey_sf[valid_gmag]**2 + 1 / KNN_FOR_ASTROLINK**2)  # Avoids diverging values and stops the total selection function from being unreasonably small
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
    chunk_n_rows = min(int(WORKING_MEMORY * (2**20) // (16 * KNN_FOR_SELECTION_FUNCTION)), n)
    batches = list(gen_batches(n, chunk_n_rows))
    num_batches = len(batches)

    # Initialize total selection function array for stars in the subsample
    print("... initializing total selection function arrays for stars in the subsample")
    nsub = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    nmw = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    total_sf_mean = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    total_sf_var = np.full_like(valid_gmag, fill_value=np.nan, dtype=np.float64)
    valid_gmag = np.where(valid_gmag)[0]  # Indices of stars with valid G-band magnitudes

    # Compute total selection function for each star in the subsample
    for i, sl in enumerate(batches):
        print(f"... computing total selection function for each star in subsample -- batch {i + 1} of {num_batches}   ", end='\r')
        # k-nearest neighbours query
        sqr_dists, idx = tree.query(xyz_stars[sl], k=KNN_FOR_SELECTION_FUNCTION, sqr_dists=True)
        del sqr_dists  # Free memory
        gc.collect()  # Force garbage collection

        # Total selection function is the posterior distribution Beta(n_sub + 1, n_mw - n_sub + 1)
        nsub_batch = subsample_sf[idx].sum(axis=1)
        nmw_batch = inverse_survey_sf[idx].sum(axis=1)
        valid_slice = valid_gmag[sl]
        nsub[valid_slice] = nsub_batch
        nmw[valid_slice] = nmw_batch
        total_sf_mean[valid_slice] = (nsub_batch + 1) / (nmw_batch + 2)  # Mean of selection function for stars in subsample
        total_sf_var[valid_slice] = (nsub_batch + 1) * (nmw_batch - nsub_batch + 1) / ((nmw_batch + 2)**2 * (nmw_batch + 3))  # Variance of selection function for stars in subsample

        # Delete temporary variables to free memory
        del idx, nsub_batch, nmw_batch
        gc.collect()
    print(f"... range of expected number of neighbours in subsample: {nsub.min():.3f} -- {nsub.max():.3f}")
    print(f"... range of expected number of neighbours in Milky Way: {nmw.min():.3f} -- {nmw.max():.3f}")
    print(f"... range of total selection function mean: {total_sf_mean.min():.3f} -- {total_sf_mean.max():.3f}")
    print(f"... range of total selection function variance: {total_sf_var.min():.3f} -- {total_sf_var.max():.3f}")

    # Save total selection function arrays for stars
    print(f"... saving total selection function arrays for stars to:")
    print(f"\t{file_nsub} ,")
    print(f"\t{file_nmw} ,")
    print(f"\t{file_total_sf_mean} , and")
    print(f"\t{file_total_sf_var} .")
    np.save(file_nsub, nsub)
    np.save(file_nmw, nmw)
    np.save(file_total_sf_mean, total_sf_mean)
    np.save(file_total_sf_var, total_sf_var)

    del nsub, nmw, total_sf_mean, total_sf_var, xyz_stars  # Free memory
    gc.collect()  # Force garbage collection

    # Also calculate the total selection function values at the centre of each HEALPix pixel for plotting
    print("... calculating total selection function for HEALPix pixels")
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
    chunk_n_rows = min(int(WORKING_MEMORY * (2**20) // (16 * KNN_FOR_SELECTION_FUNCTION)), npix)
    batches = list(gen_batches(npix, chunk_n_rows))
    num_batches = len(batches)

    # Initialize arrays for HEALPix pixels
    print("... initializing total selection function arrays for HEALPix pixels")
    total_sf_mean_healpix = np.empty(npix)
    total_sf_var_healpix = np.empty(npix)

    # Compute m10 for each HEALPix pixel as median G of neighbors with <11 transits
    for i, sl in enumerate(batches):
        print(f"... computing total selection function for HEALPix pixels -- batch {i + 1} of {num_batches}   ", end='\r')
        # k-nearest neighbours query
        _, idx = tree.query(xyz_healpix[sl], k=KNN_FOR_SELECTION_FUNCTION, sqr_dists=True)

        # Total selection function is the posterior distribution Beta(n_sub + 1, n_mw - n_sub + 1)
        nsub_healpix_batch = subsample_sf[idx].sum(axis=1)
        nmw_healpix_batch = inverse_survey_sf[idx].sum(axis=1)
        total_sf_mean_healpix[sl] = (nsub_healpix_batch + 1) / (nmw_healpix_batch + 2)  # Mean of selection function for HEALPix pixels
        total_sf_var_healpix[sl] = (nsub_healpix_batch + 1) * (nmw_healpix_batch - nsub_healpix_batch + 1) / ((nmw_healpix_batch + 2)**2 * (nmw_healpix_batch + 3))  # Variance of selection function for HEALPix pixels

        # Delete temporary variables to free memory
        del _, idx, nsub_healpix_batch, nmw_healpix_batch
        gc.collect()
    
    del tree, xyz_healpix, valid_gmag, subsample_sf, inverse_survey_sf  # Free memory
    gc.collect()  # Force garbage collection

    # Save total selection function arrays for healpix pixels
    print(f"... saving total selection function arrays for HEALPix pixels to:")
    print(f"\t{file_total_sf_mean_healpix} , and")
    print(f"\t{file_total_sf_var_healpix} .\n")
    np.save(file_total_sf_mean_healpix, total_sf_mean_healpix)
    np.save(file_total_sf_var_healpix, total_sf_var_healpix)
    del total_sf_mean_healpix, total_sf_var_healpix  # Free memory
    gc.collect()  # Force garbage collection

def plot_total_selection_function_for_subsample(overwrite=False):
    """
    Plot the limiting G-band magnitude across the sky using HEALPix.
    """
    # Check if plots already exists
    file_total_sf_mean_path = os.path.join(FIGURES_PATH, "total_selection_function_mean.png")
    file_total_sf_var_path = os.path.join(FIGURES_PATH, "total_selection_function_var.png")
    if os.path.exists(file_total_sf_mean_path) and os.path.exists(file_total_sf_var_path) and not overwrite:
        print(f"Total selection function mean and variance plots already exist at:")
        print(f"\t{file_total_sf_mean_path} and")
        print(f"\t{file_total_sf_var_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting total selection function on the sky...")

    # Load total selection function for HEALPix pixels
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

    # Create a Mollweide projection plot of the total selection function variance
    plt.figure(figsize=(12, 6))
    projview(
        total_sf_var,
        coord=["G"],
        nest=True,
        unit=r"Total selection function variance, $\mathrm{Var}[S_{\mathrm{total}}]$",
        cb_orientation="horizontal",
        #min=0,
        #max=1,
        cmap="magma",
        projection_type="mollweide",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_total_sf_var_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    
    print(f"... saved mollview plot to {file_total_sf_var_path}.\n")


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
    r = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_med_geo.npy"))[subsample_mask]  # (n,) in pc
    lo, high = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_lo_high_geo.npy"))[subsample_mask].T  # each (n,) in pc
    ra, dec = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_equatorial_coordinates.npy"))[subsample_mask].T  # each (n,) in degrees
    dra, ddec = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_astrometric_errors.npy"))[subsample_mask, :2].T  # each (n,) in degrees
    
    dr = (high - lo) / 2
    variances = np.column_stack([
        dr**2,  # Variance in LOS
        (np.cos(np.deg2rad(dec)) * np.deg2rad(dra))**2,  # Variance in RA
        np.deg2rad(ddec)**2 # Variance in Dec
    ])
    del subsample_mask, lo, high, dr, ra, dec, dra, ddec  # Free memory
    gc.collect()  # Force garbage collection

    # Fit model using a grid search for r_{1/2}
    print("... fitting r_{1/2} to get globally isotropic spatial uncertainties")
    bounds = (1, 1000)  # Initial guess for r_{1/2} in pc
    result = minimize_scalar(
        lambda r_half: average_sym_kl_contracted(r_half, variances, r),
        bounds=bounds,
        method='bounded',
        options={'xatol': 1.0}      # stop when r_half is within 1 pc
    )
    r_half = result.x  # Best fit characteristic scale r_{1/2} in pc
    print(f"... best fit r_half = {r_half:.3f} pc, with loss = {result.fun:.3f}")

    # Save the best fit r_{1/2}
    print(f"... saving best fit r_half to {file_path_r_half} (shape: {r_half.shape})")
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
    print(f"... saving contracted distance uncertainties to {file_path_delta_fr} (shape: {dfr.shape})")
    np.save(file_path_delta_fr, dfr)
    del dfr  # Free memory
    gc.collect()  # Force garbage collection

@njit()
def average_sym_kl_contracted(r_half, variances, r):
    """
    Compute the average symmetrized KL divergence between propagated Gaia-like
    spherical coordinate uncertainties and an optimal isotropic Gaussian under
    a contracted distance metric f(r) = r_half * arctan(r / r_half).

    Parameters:
        r_half : float
            Contraction scale parameter r_{1/2} (in pc).
        variances : np.ndarray of shape (N, 3)
            Each row contains uncertainties: (delta_r^2, delta_l*^2, delta_b^2)
            where delta_l* = cos(b) * delta_l in radians.
        r : np.ndarray of shape (N,)
            Radial distances (in pc) for each source.

    Returns:
        alpha_opt : float
            Optimal scalar variance alpha.
        avg_kl_sym : float
            Average symmetrized KL divergence in contracted space.
    """
    # Unpack uncertainties of observables
    var_r, var_lstar, var_b = variances.T

    # Contracted distance and its derivative
    f_r = r_half * np.arctan(r / r_half)
    f_prime = (r_half**2) / (r_half**2 + r**2)

    # First-order propagated variances (diagonal)
    var_los = f_prime**2 * var_r              # LOS direction
    var_p1 = f_r**2 * var_lstar              # horizontal tangential
    var_p2 = f_r**2 * var_b                  # vertical tangential

    # Combine into diagonal covariance matrix for each source
    tr = var_los + var_p1 + var_p2
    tr_inv = 1.0 / var_los + 1.0 / var_p1 + 1.0 / var_p2

    # Average symmetrized KL divergence
    avg_kl_sym = np.mean(np.sqrt(tr * tr_inv)) - 3

    print("\t... r_{1/2}:", r_half, "loss:", avg_kl_sym)

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
    lo, high = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "bailerjones_r_lo_high_geo.npy"))[subsample_mask].T  # each (N,) in pc
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

def construct_cartesian_coordinates_for_subsample(overwrite=False):
    """
    Calculate the Cartesian-like coordinates for the subsample.
    """
    # Check if Cartesian coordinates already exist
    file_cartesian_coordinates = os.path.join(SUBSAMPLE_PATH, "contracted_cartesian_coordinates.npy")
    if os.path.exists(file_cartesian_coordinates) and not overwrite:
        print(f"Contracted Cartesian coordinates already exist at:\n\t{file_cartesian_coordinates} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Calculating contracted Cartesian coordinates for subsample...")

    # Load the contracted positions and velocities
    print("... loading required arrays")
    positions = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_positions.npy"))  # (N, 3)
    velocities = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_velocities.npy"))  # (N, 3)
    delta_pos = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_position_uncertainties.npy"))  # (N,)
    delta_vel = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_velocity_uncertainties.npy"))  # (N,)

    # Calculate scaling factor for positions
    alpha_pos = np.median(delta_pos)  # Calculate scaling factor
    print(f"... scaling factor for positions: {alpha_pos:.3f}")
    positions /= alpha_pos  # Scale positions

    # Calculate scaling factor for velocities
    alpha_vel = np.median(delta_vel)  # Calculate scaling factor
    print(f"... scaling factor for velocities: {alpha_vel:.3f}")
    velocities /= alpha_vel  # Scale velocities

    # Concatenate positions and velocities to form Cartesian coordinates
    print("... constructing Cartesian-like coordinates")
    cartesian_coordinates = np.concatenate([positions, velocities], axis=1)  # shape (N, 6)

    # Save Cartesian coordinates
    print(f"... saving Cartesian coordinates to {file_cartesian_coordinates} (shape: {cartesian_coordinates.shape}).\n")
    np.save(file_cartesian_coordinates, cartesian_coordinates)


# === Apply AstroLink to subsample and plot of cluster properties ===
def apply_astrolink_to_subsample(overwrite=True):
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
    cartesian_coordinates = np.load(os.path.join(SUBSAMPLE_PATH, "contracted_cartesian_coordinates.npy"))  # (N, 6)
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N,)
    total_sf_mean = np.load(os.path.join(SUBSAMPLE_PATH, "total_selection_function_mean.npy"))[subsample_mask]  # (N,)
    del subsample_mask  # Free memory
    gc.collect()  # Force garbage collection

    # Initialize AstroLink
    print("... initializing AstroLink object")
    clusterer = AstroLink(
        P=cartesian_coordinates,
        d_intrinsic=5,
        weights=total_sf_mean,
        k_den=KNN_FOR_ASTROLINK,
        adaptive=0,
        S=SIGMA_FOR_ASTROLINK,
        workers=PARALLEL_WORKERS,
        verbose=0
    )
    del cartesian_coordinates, total_sf_mean  # Free memory
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

    # Save the clustering output
    print(f"... saving AstroLink clustering output to {file_astrolink_object}.\n")
    io.saveAstroLinkObject(clusterer, file_astrolink_object)

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
    clusterer = io.loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    
    # Plot the prominence model fit
    fig, ax = plt.subplots(figsize=(8, 6))
    h, _, _, _, _ = visualize.prominenceModel(clusterer, ax=ax, cutoffKwargs={'alpha': 0.0})

    # Add vertical lines at various significance levels
    #offset = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])  # small offset to the left
    for i, sig, in enumerate(np.linspace(3, 5, 5)):
        prom = beta.isf(norm.sf(sig), clusterer.pFit[0], clusterer.pFit[1])  # Inverse survival function for beta distribution
        ax.axvline(x=prom, color=f"C{i}", linestyle='--', linewidth=2)
        ax.text(prom, 0.75 * h.max(), f"S = {sig:.1f}",
            color=f"C{i}", fontsize=10, rotation=90, ha='right', va='top')

    # Convert y-axis to logarithmic scale
    ax.set_xlim(0, beta.isf(norm.sf(7), clusterer.pFit[0], clusterer.pFit[1]))
    ax.set_ylim(h[h > 0].min() * 0.5, ax.get_ylim()[1])  # Set y-axis limits to avoid zero and very high values
    ax.set_yscale('log')  # Set y-axis to logarithmic scale

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_prominence_model_fit_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved prominence model fit plot to {file_prominence_model_fit_path}.\n")

def plot_number_of_clusters_vs_significance(overwrite=False):
    """
    Plot the number of clusters vs significance from AstroLink.
    """
    # Check if plots already exist
    file_clusters_vs_significance_path = os.path.join(FIGURES_PATH, "n_clusters_vs_significance.png")
    if os.path.exists(file_clusters_vs_significance_path) and not overwrite:
        print(f"Clusters vs significance plot already exists at:\n\t{file_clusters_vs_significance_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting number of clusters vs significance...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = io.loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    
    # Plot the number of clusters vs significance
    fig, ax = plt.subplots(figsize=(8, 6))

    significances = np.linspace(3, 10, 71)  # Significance levels from 3 to 10
    num_clusters = []
    for sig in significances:
        clusterer.S = sig
        clusterer.extract_clusters()
        num_clusters.append(len(clusterer.clusters) - 1)  # Exclude the background cluster

    ax.loglog(significances, num_clusters, color='C0')
    plt.grid(True, which="both", ls="-")
    ax.set_xlabel(r"Significance, $S$")
    ax.set_ylabel(r"Number of clusters, $N(>S)$")

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_clusters_vs_significance_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    print(f"... saved clusters vs significance plot to {file_clusters_vs_significance_path}.\n")

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
    clusterer = io.loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    clusterer.S = 3.8
    clusterer.extract_clusters()
    print(f"... found {len(clusterer.clusters) - 1} clusters at S={clusterer.S} in the clustering output")

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

    # Create a Mollweide projection plot and plot clusters on the sky
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

    # Cycle through the clusters and plot them
    for i, clst in enumerate(clusterer.clusters[1:]):
        clusterMembers = clusterer.ordering[clst[0]:clst[1]]
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
    clusterer = io.loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    clusterer.S = 3.8
    clusterer.extract_clusters()

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


# === Compare clustering output to Hunt & Reffert (2024) ===
def prepare_for_Hunt2024_comparison(overwrite=False):
    """
    Prepare the data for comparison with Hunt & Reffert (2024).
    """
    # Check if files already exist
    file_path_source_ids = os.path.join(CLUSTERING_PATH, "hunt24_source_ids.npy")
    file_path_members_mask = os.path.join(CLUSTERING_PATH, "hunt24_members_mask.npy")
    file_path_members_cluster_ids = os.path.join(CLUSTERING_PATH, "hunt24_members_cluster_ids.npy")
    file_path_members_probs = os.path.join(CLUSTERING_PATH, "hunt24_members_probs.npy")
    file_path_clusters_names = os.path.join(CLUSTERING_PATH, "hunt24_clusters_names.npy")
    file_path_clusters_types = os.path.join(CLUSTERING_PATH, "hunt24_clusters_types.npy")
    file_path_clusters_snr = os.path.join(CLUSTERING_PATH, "hunt24_clusters_snr.npy")
    if (os.path.exists(file_path_source_ids) and
        os.path.exists(file_path_members_mask) and
        os.path.exists(file_path_members_cluster_ids) and
        os.path.exists(file_path_members_probs) and
        os.path.exists(file_path_clusters_names) and
        os.path.exists(file_path_clusters_types) and
        os.path.exists(file_path_clusters_snr)) and not overwrite:
        print("Hunt & Reffert (2024) reduced data already exists at:")
        print(f"\t{file_path_source_ids} ,")
        print(f"\t{file_path_members_mask} ,")
        print(f"\t{file_path_members_cluster_ids} ,")
        print(f"\t{file_path_members_probs} ,")
        print(f"\t{file_path_clusters_names} ,")
        print(f"\t{file_path_clusters_types} , and")
        print(f"\t{file_path_clusters_snr} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with Hunt & Reffert (2024)...")

    # Read the clusters.dat.gz file
    print("... loading clusters.dat.gz data")
    readme_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "ReadMe")
    clusters_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "clusters.dat.gz")
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
    H24_members_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "members.dat.gz")
    df_members = load_cds_table(readme_path, H24_members_path)

    # Save the cluster IDs and probabilities
    print("... saving member information")
    H24_members_source_ids = df_members['GaiaDR3'].to_numpy()  # Source IDs of the members
    H24_members_cluster_ids = df_members["ID"].to_numpy()  # Cluster IDs
    H24_members_probs = df_members["Prob"].to_numpy()  # Membership probabilities
    np.save(file_path_source_ids, H24_members_source_ids)  # Save source IDs
    np.save(file_path_members_cluster_ids, H24_members_cluster_ids)
    np.save(file_path_members_probs, H24_members_probs)
    del df_members, H24_members_cluster_ids, H24_members_probs  # Free memory
    gc.collect()  # Force garbage collection

    # Load Gaia DR3 source_ids
    print("... loading Gaia DR3 source IDs")
    gdr3_source_ids = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_source_ids.npy"))  # (N,)

    # Save the membership mask
    print("... making membership mask")
    indices = np.searchsorted(gdr3_source_ids, H24_members_source_ids) # Assumes gdr3_source_ids is sorted
    H24_members_mask = np.zeros_like(gdr3_source_ids, dtype=np.bool_)  # Create a mask of the same shape as gdr3_source_ids
    H24_members_mask[indices] = True  # Set the indices of the members to True
    del gdr3_source_ids, H24_members_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Save the membership mask
    print(f"... saving membership mask to {file_path_members_mask} (shape: {H24_members_mask.shape}).\n")
    np.save(file_path_members_mask, H24_members_mask)
    del H24_members_mask  # Free memory
    gc.collect()  # Force garbage collection

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

def compare_to_Hunt2024(overwrite=True):
    """
    Compare the clustering output to the Hunt & Reffert (2024).
    """
    # Check if comparison results already exist
    file_path_cluster_rpje = os.path.join(CLUSTERING_PATH, "hunt24_rpje.npy")
    file_path_best_match_astrolink_clusters = os.path.join(CLUSTERING_PATH, "hunt24_best_match_astrolink_clusters.npy")
    file_path_number_of_astrolink_clusters_per_sig = os.path.join(CLUSTERING_PATH, "hunt24_number_of_astrolink_clusters_per_sig.npy")
    if (os.path.exists(file_path_cluster_rpje) and
        os.path.exists(file_path_best_match_astrolink_clusters) and
        os.path.exists(file_path_number_of_astrolink_clusters_per_sig)) and not overwrite:
        print("Hunt & Reffert (2024) comparison results already exist at:")
        print(f"\t{file_path_cluster_rpje} ,")
        print(f"\t{file_path_best_match_astrolink_clusters} , and")
        print(f"\t{file_path_number_of_astrolink_clusters_per_sig} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to Hunt & Reffert (2024)...")

    # Load required arrays
    print("... loading required arrays for comparison")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N_gdr3,)
    H24_members_source_ids = np.load(os.path.join(CLUSTERING_PATH, "hunt24_source_ids.npy"))  # (N,)
    H24_members_mask = np.load(os.path.join(CLUSTERING_PATH, "hunt24_members_mask.npy"))  # (N_gdr3,)
    H24_members_cluster_ids = np.load(os.path.join(CLUSTERING_PATH, "hunt24_members_cluster_ids.npy"))  # (N,)
    H24_members_probs = np.load(os.path.join(CLUSTERING_PATH, "hunt24_members_probs.npy"))  # (N,)

    # Get Hunt+2024 cluster IDs for this subsample (multiple columns since stars can be in multiple Hunt+2024 clusters)
    print("... getting Hunt+2024 cluster IDs and membership probabilities for the subsample in this work")
    max_appearances = np.unique(H24_members_source_ids, return_counts=True)[1].max()
    max_H24_cluster_ID = H24_members_cluster_ids.max()
    H24_members_cluster_ids_gdr3 = np.full((H24_members_mask.size, max_appearances), max_H24_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_H24_cluster_ID + 1, representing no cluster
    H24_members_probs_gdr3 = np.zeros((H24_members_mask.size, max_appearances), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
    H24_members_mask_where = np.where(H24_members_mask)[0]  # Use indices of members in the full catalogue from now on to do efficient slicing/indexing
    
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
        H24_members_probs_gdr3[H24_members_mask_where[num_count_bool], :num_count] = members_probs

    H24_members_cluster_ids_subsample = H24_members_cluster_ids_gdr3[subsample_mask]
    H24_members_probs_subsample = H24_members_probs_gdr3[subsample_mask]
    del subsample_mask, H24_members_source_ids, H24_members_mask, H24_members_cluster_ids_gdr3, H24_members_probs_gdr3  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the total sum of probabilities for each Hunt+2024 cluster
    print("... pre-computing the total sum of probabilities for each Hunt+2024 cluster")
    H24_cluster_probability_sums_total = np.bincount(H24_members_cluster_ids, 
                                        weights=H24_members_probs,
                                        minlength=max_H24_cluster_ID + 1)  # (N_clusters,)
    del H24_members_cluster_ids, H24_members_probs  # Free memory
    gc.collect()  # Force garbage collection

    # Pre-compute the sum of probabilities for each Hunt+2024 cluster in the overlap with the subsample
    print("... pre-computing the total sum of probabilities for each Hunt+2024 cluster in the subsample of this work")
    H24_cluster_probability_sums_overlap = np.bincount(H24_members_cluster_ids_subsample.ravel(),
                                        weights=H24_members_probs_subsample.ravel(),
                                        minlength=max_H24_cluster_ID + 1)[:max_H24_cluster_ID + 1]  # (N_clusters,)

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = io.loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    ordering = clusterer.ordering  # avoid sending the whole clusterer to workers

    
    # Put large arrays into shared memory
    print("... putting large arrays into shared memory")
    def to_shm(arr, dtype):
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=dtype, buffer=shm.buf)
        np.copyto(shm_arr, arr)
        return shm, arr.shape, dtype

    shm_ids, shape_ids, dtype_ids = to_shm(H24_members_cluster_ids_subsample, np.int64)
    shm_probs, shape_probs, dtype_probs = to_shm(H24_members_probs_subsample, np.float32)
    shm_ordering, shape_ordering, dtype_ordering = to_shm(ordering, ordering.dtype.type)
    
    # Calculate the RPJE values for each significance level
    whichClusters = -np.ones((SIGMA_THRESHOLDS_FOR_COMPARISONS.size, H24_cluster_probability_sums_total.size, 2), dtype=np.int64)  # (N_clusters, 2) to store AstroLink clusters (start, end) pairs
    RPJE = np.zeros((SIGMA_THRESHOLDS_FOR_COMPARISONS.size, H24_cluster_probability_sums_total.size, 4), dtype=np.float32)  # (N_clusters, 4) to store RPJE values
    num_astrolink_clusters = np.zeros(SIGMA_THRESHOLDS_FOR_COMPARISONS.size, dtype=np.int64)  # Number of AstroLink clusters for each significance level
    
    # Loop over significance values
    for k, significance in enumerate(SIGMA_THRESHOLDS_FOR_COMPARISONS):
        print(f"... calculating recovery, purity, Jaccard-index, and evidence values at significance level S={significance:.1f}   ", end = '\r')
        # Extract clusters at the current significance level
        clusterer.S = significance
        clusterer.extract_clusters()

        # Track the number of clusters at the current significance level
        num_astrolink_clusters[k] = clusterer.clusters.shape[0] - 1  # Exclude the background cluster

        # Loop over AstroLink clusters in parallel
        max_workers = min(8, PARALLEL_WORKERS) # Use limited number of workers because this process is memory intensive
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, (start, end) in enumerate(clusterer.clusters[1:]):
                futures.append(
                    executor.submit(process_astrolink_cluster,
                                    start, end,
                                    i, num_astrolink_clusters[k],
                                    H24_cluster_probability_sums_total,
                                    H24_cluster_probability_sums_overlap,
                                    max_H24_cluster_ID,
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
                better_matches = RPJE_cluster[:, 2] > RPJE[k, unique_ids, 2]
                which_better_matches = unique_ids[better_matches]
                whichClusters[k, which_better_matches] = start, end
                RPJE[k, which_better_matches] = RPJE_cluster[better_matches]
    
    # Clean up shared memory
    print("... cleaning up shared memory                                                                                                  ")
    shm_ids.close(); shm_ids.unlink()
    shm_probs.close(); shm_probs.unlink()
    shm_ordering.close(); shm_ordering.unlink()
    
    del clusterer, ordering, H24_members_cluster_ids_subsample, H24_members_probs_subsample, H24_cluster_probability_sums_total, H24_cluster_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_best_match_astrolink_clusters, whichClusters)
    np.save(file_path_cluster_rpje, RPJE)
    np.save(file_path_number_of_astrolink_clusters_per_sig, num_astrolink_clusters)
    del whichClusters, RPJE, num_astrolink_clusters  # Free memory
    gc.collect()  # Force garbage collection

def process_astrolink_cluster(start, end,
                              i, total_astrolink_clusters,
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

    # Get number of members in the AstroLink cluster when adjusted for the intersection with Hunt+2024 clusters
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
    RPJE = np.zeros((unique_ids.size, 4), dtype=np.float32)

    # Probability mass of the clusters
    Prob_sums = cluster_probability_sums_overlap[unique_ids]

    # Pre-calculate a term to be used twice
    union_in_overlap = N_i + Prob_sums - M_sums

    # Calculate and store the recovery, purity, Jaccard index, and evidence values
    RPJE[:, 0] = M_sums / Prob_sums
    RPJE[:, 1] = M_sums / N_i
    RPJE[:, 2] = M_sums / union_in_overlap
    RPJE[:, 3] = union_in_overlap / (end - start + cluster_probability_sums_total[unique_ids] - M_sums)

    return RPJE

def plot_evidence_weighted_Hunt2024_comparison_results(overwrite=False):
    """
    Plot the results of the comparison between clustering output and Hunt & Reffert (2024).
    """
    # Check if plot already exists
    file_path = os.path.join(FIGURES_PATH, "Hunt2024_evidence_weighted_comparison_results.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Hunt & Reffert (2024) evidence-weighted comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Hunt & Reffert (2024) evidence-weighted comparison results...")

    # Load the comparison results
    print("... loading comparison results")
    RPJE = np.load(os.path.join(CLUSTERING_PATH, "hunt24_rpje.npy"))  # (N_sigmas, N_clusters, 4)
    #num_astrolink_clusters = np.load(os.path.join(CLUSTERING_PATH, "number_of_astrolink_clusters.npy"))  # (N_sigmas,)

    # Load the Hunt & Reffert (2024) cluster types
    print("... loading Hunt & Reffert (2024) cluster types")
    H24_cluster_types = np.load(os.path.join(CLUSTERING_PATH, "hunt24_clusters_types.npy"), allow_pickle=True)  # (N_clusters,)

    # Weight the recovery, purity, and Jaccard index by the evidence
    print("... weighting the recovery, purity, and Jaccard index by the evidence")
    RPJE[..., 0] *= RPJE[..., 3]  # Evidence-weighted recovery
    RPJE[..., 1] *= RPJE[..., 3]  # Evidence-weighted purity
    RPJE[..., 2] *= RPJE[..., 3]  # Evidence-weighted Jaccard index

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the recovery, purity, and Jaccard index for each significance level
    print("... plotting the recovery, purity, and Jaccard index vs significance level for all clusters")
    mask = (H24_cluster_types != 'r') * (H24_cluster_types != 'd')
    sum_of_evidence_weights = np.sum(RPJE[:, mask, 3], axis=1)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
            np.sum(RPJE[:, mask, 0], axis=1) / sum_of_evidence_weights,
            color='k', linestyle='dotted', linewidth=1.5,
            label='Recovery (o,m,g)')
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
            np.sum(RPJE[:, mask, 1], axis=1) / sum_of_evidence_weights,
            color='k', linestyle='dashed', linewidth=1.5,
            label='Purity (o,m,g)')
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
            np.sum(RPJE[:, mask, 2], axis=1) / sum_of_evidence_weights,
            color='k', linestyle='solid', linewidth=1.5,
            label='Jaccard index (o,m,g)')

    print('Best fit S=', SIGMA_THRESHOLDS_FOR_COMPARISONS[np.argmax(np.sum(RPJE[:, mask, 2], axis=1) / sum_of_evidence_weights)])

    # Plot the recovery, purity, and Jaccard index for each significance level for each cluster type
    print("... plotting the recovery, purity, and Jaccard index vs significance level for each cluster type")
    cluster_type_and_colour = dict(zip(['o', 'm', 'g', 'd', 'r'], ['C0', 'C2', 'C1', 'C4', 'C3']))
    for cluster_type, type_colour in cluster_type_and_colour.items():
        mask = H24_cluster_types == cluster_type
        sum_of_evidence_weights = np.sum(RPJE[:, mask, 3], axis=1)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
                np.sum(RPJE[:, mask, 2], axis=1) / sum_of_evidence_weights,
                color=type_colour, linestyle='solid', linewidth=0.75, alpha=0.75,
                label=f"Jaccard index ({cluster_type})")

    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Significance Level")
    ax.set_ylabel("Comparison Statistic")
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)



# === Compare clustering output to the Unified Cluster Catalogue ===
def prepare_for_UCC_comparison(overwrite=False):
    """
    Prepare the data for comparison with the Unified Cluster Catalogue.
    """
    # Check if files already exist
    file_path_clusters_names = os.path.join(CLUSTERING_PATH, "ucc_clusters_names.npy")
    file_path_clusters_quality_class = os.path.join(CLUSTERING_PATH, "ucc_clusters_quality_class.npy")
    file_path_members_cluster_names = os.path.join(CLUSTERING_PATH, "ucc_members_cluster_names.npy")
    file_path_members_probs = os.path.join(CLUSTERING_PATH, "ucc_members_probs.npy")
    file_path_source_ids = os.path.join(CLUSTERING_PATH, "ucc_source_ids.npy")
    file_path_members_mask = os.path.join(CLUSTERING_PATH, "ucc_members_mask.npy")

    # Skip processing if all merged output files already exist
    all_exist = all([
        os.path.exists(file_path_clusters_names),
        os.path.exists(file_path_clusters_quality_class),
        os.path.exists(file_path_members_cluster_names),
        os.path.exists(file_path_members_probs),
        os.path.exists(file_path_source_ids),
        os.path.exists(file_path_members_mask)
    ])
    if all_exist and not overwrite:
        print("Unified Cluster Catalogue reduced data already exists at:")
        print(f"\t{file_path_clusters_names} ,")
        print(f"\t{file_path_clusters_quality_class} ,")
        print(f"\t{file_path_members_cluster_names} ,")
        print(f"\t{file_path_members_probs} ,")
        print(f"\t{file_path_source_ids} , and")
        print(f"\t{file_path_members_mask} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Preparing data for comparison with the Unified Cluster Catalogue...")

    # Read the UCC_cat.csv file
    print("... loading UCC_cat.csv data")
    cat_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC_cat.csv")
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
    UCC_members_path = os.path.join(AUXILLARY_CATALOGUES_PATH, "UCC_members.parquet")
    df_members = pd.read_parquet(UCC_members_path)

    # Save the cluster names for each of the members
    print("... saving member cluster names")
    UCC_members_cluster_names = df_members["name"].to_numpy().astype(np.str_)  # Cluster names
    np.save(file_path_members_cluster_names, UCC_members_cluster_names)
    del UCC_members_cluster_names  # Free memory
    gc.collect()  # Force garbage collection

    # Saving membership probabilities
    print("... saving member membership probabilities")
    UCC_members_probs = df_members["probs"].to_numpy()  # Membership probabilities
    np.save(file_path_members_probs, UCC_members_probs)
    del UCC_members_probs  # Free memory
    gc.collect()  # Force garbage collection

    # Save the GDR3 source ids
    print("... saving member source ids")
    UCC_members_source_ids = df_members['Source'].to_numpy()  # Source IDs of the members
    np.save(file_path_source_ids, UCC_members_source_ids)  # Save source IDs
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
    del gdr3_source_ids, UCC_members_source_ids, indices  # Free memory
    gc.collect()  # Force garbage collection

    # Save the membership mask
    print(f"... saving membership mask to {file_path_members_mask} (shape: {UCC_members_mask.shape}).\n")
    np.save(file_path_members_mask, UCC_members_mask)
    del UCC_members_mask  # Free memory
    gc.collect()  # Force garbage collection

def compare_to_UCC(overwrite=True):
    """
    Compare the clustering output to the Unified Cluster Catalogue.
    """
    # Check if comparison results already exist
    file_path_cluster_rpje = os.path.join(CLUSTERING_PATH, "ucc_rpje.npy")
    file_path_best_match_astrolink_clusters = os.path.join(CLUSTERING_PATH, "ucc_best_match_astrolink_clusters.npy")
    file_path_number_of_astrolink_clusters_per_sig = os.path.join(CLUSTERING_PATH, "ucc_number_of_astrolink_clusters_per_sig.npy")

    # Skip processing if all merged output files already exist
    all_exist = (os.path.exists(file_path_cluster_rpje) and
                  os.path.exists(file_path_best_match_astrolink_clusters) and
                  os.path.exists(file_path_number_of_astrolink_clusters_per_sig))
    if all_exist and not overwrite:
        print("Unified Cluster Catalogue comparison results already exist at:")
        print(f"\t{file_path_cluster_rpje} ,")
        print(f"\t{file_path_best_match_astrolink_clusters} , and")
        print(f"\t{file_path_number_of_astrolink_clusters_per_sig} .")
        print("Use overwrite=True to force recomputation.\n")
        return
    print("Comparing clustering output to the Unified Cluster Catalogue...")

    # Load required arrays
    print("... loading required arrays for comparison")
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "subsample_mask.npy"))  # (N_gdr3,)
    UCC_members_source_ids = np.load(os.path.join(CLUSTERING_PATH, "ucc_source_ids.npy"))  # (N,)
    UCC_members_mask = np.load(os.path.join(CLUSTERING_PATH, "ucc_members_mask.npy"))  # (N_gdr3,)
    UCC_members_cluster_names = np.load(os.path.join(CLUSTERING_PATH, "ucc_members_cluster_names.npy"))  # (N,)
    UCC_members_probs = np.load(os.path.join(CLUSTERING_PATH, "ucc_members_probs.npy"))  # (N,)

    # Get UCC cluster IDs for this subsample (multiple columns since stars can be in multiple UCC clusters)
    print("... getting UCC cluster IDs and membership probabilities for the subsample in this work")
    max_appearances = np.unique(UCC_members_source_ids, return_counts=True)[1].max()
    unique_UCC_members_cluster_names, UCC_members_cluster_ids = np.unique(UCC_members_cluster_names, return_inverse=True)
    max_UCC_cluster_ID = UCC_members_cluster_ids.max()
    UCC_members_cluster_ids_gdr3 = np.full((UCC_members_mask.size, max_appearances), max_UCC_cluster_ID + 1, dtype=np.int64)  # (N,) Initialize with max_UCC_cluster_ID + 1, representing no cluster
    UCC_members_probs_gdr3 = np.zeros((UCC_members_mask.size, max_appearances), dtype=np.float32)  # (N,) Initialize with 0, representing zero membership probability
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
        UCC_members_probs_gdr3[UCC_members_mask_where[num_count_bool], :num_count] = members_probs

    UCC_members_cluster_ids_subsample = UCC_members_cluster_ids_gdr3[subsample_mask]
    UCC_members_probs_subsample = UCC_members_probs_gdr3[subsample_mask]
    del subsample_mask, UCC_members_source_ids, UCC_members_cluster_ids_gdr3, UCC_members_probs_gdr3, UCC_members_mask_where, unique_UCC_source_ids, indices, counts  # Free memory
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
                                        weights=UCC_members_probs_subsample.ravel(),
                                        minlength=max_UCC_cluster_ID + 1)[:max_UCC_cluster_ID + 1]  # (N_clusters,)

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = io.loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    ordering = clusterer.ordering  # avoid sending the whole clusterer to workers
    
    # Put large arrays into shared memory
    print("... putting large arrays into shared memory")
    def to_shm(arr, dtype):
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=dtype, buffer=shm.buf)
        np.copyto(shm_arr, arr)
        return shm, arr.shape, dtype

    shm_ids, shape_ids, dtype_ids = to_shm(UCC_members_cluster_ids_subsample, np.int64)
    shm_probs, shape_probs, dtype_probs = to_shm(UCC_members_probs_subsample, np.float32)
    shm_ordering, shape_ordering, dtype_ordering = to_shm(ordering, ordering.dtype.type)
    
    # Calculate the RPJE values for each significance level
    whichClusters = -np.ones((SIGMA_THRESHOLDS_FOR_COMPARISONS.size, UCC_cluster_probability_sums_total.size, 2), dtype=np.int64)  # (N_clusters, 2) to store AstroLink clusters (start, end) pairs
    RPJE = np.zeros((SIGMA_THRESHOLDS_FOR_COMPARISONS.size, UCC_cluster_probability_sums_total.size, 4), dtype=np.float32)  # (N_clusters, 4) to store RPJE values
    num_astrolink_clusters = np.zeros(SIGMA_THRESHOLDS_FOR_COMPARISONS.size, dtype=np.int64)  # Number of AstroLink clusters for each significance level
    
    # Loop over significance values
    for k, significance in enumerate(SIGMA_THRESHOLDS_FOR_COMPARISONS):
        print(f"... calculating recovery, purity, Jaccard-index, and evidence values at significance level S={significance:.1f}   ", end = '\r')
        # Extract clusters at the current significance level
        clusterer.S = significance
        clusterer.extract_clusters()

        # Track the number of clusters at the current significance level
        num_astrolink_clusters[k] = clusterer.clusters.shape[0] - 1  # Exclude the background cluster

        # Loop over AstroLink clusters in parallel
        max_workers = min(4, PARALLEL_WORKERS) # Use limited number of workers because this process is memory intensive
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, (start, end) in enumerate(clusterer.clusters[1:]):
                futures.append(
                    executor.submit(process_astrolink_cluster,
                                    start, end,
                                    i, num_astrolink_clusters[k],
                                    UCC_cluster_probability_sums_total,
                                    UCC_cluster_probability_sums_overlap,
                                    max_UCC_cluster_ID,
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
                better_matches = RPJE_cluster[:, 2] > RPJE[k, unique_ids, 2]
                which_better_matches = unique_ids[better_matches]
                whichClusters[k, which_better_matches] = start, end
                RPJE[k, which_better_matches] = RPJE_cluster[better_matches]
    
    # Clean up shared memory
    print("... cleaning up shared memory                                                                                                  ")
    shm_ids.close(); shm_ids.unlink()
    shm_probs.close(); shm_probs.unlink()
    shm_ordering.close(); shm_ordering.unlink()

    del clusterer, ordering, UCC_members_cluster_ids_subsample, UCC_members_probs_subsample, UCC_cluster_probability_sums_total, UCC_cluster_probability_sums_overlap  # Free memory
    gc.collect()  # Force garbage collection

    # Save the results
    print("... saving comparison results.\n")
    np.save(file_path_best_match_astrolink_clusters, whichClusters)
    np.save(file_path_cluster_rpje, RPJE)
    np.save(file_path_number_of_astrolink_clusters_per_sig, num_astrolink_clusters)
    del whichClusters, RPJE, num_astrolink_clusters  # Free memory
    gc.collect()  # Force garbage collection

def plot_evidence_weighted_UCC_comparison_results(overwrite=True):
    """
    Plot the results of the comparison between clustering output and the Unified Cluster Catalogue.
    """
    # Check if plot already exists
    file_path = os.path.join(FIGURES_PATH, "UCC_evidence_weighted_comparison_results.png")
    if os.path.exists(file_path) and not overwrite:
        print(f"Unified Cluster Catalogue evidence-weighted comparison results plot already exists at:\n\t{file_path} .")
        print("Use overwrite=True to force replotting.\n")
        return
    print("Plotting Unified Cluster Catalogue evidence-weighted comparison results...")

    # Load the comparison results
    print("... loading comparison results")
    RPJE = np.load(os.path.join(CLUSTERING_PATH, "ucc_rpje.npy"))  # (N_sigmas, N_clusters, 4)
    num_astrolink_clusters = np.load(os.path.join(CLUSTERING_PATH, "ucc_number_of_astrolink_clusters_per_sig.npy"))  # (N_sigmas,)

    # Load the Unified Cluster Catalogue cluster types
    print("... loading Unified Cluster Catalogue cluster names and quality classes")
    UCC_clusters_names = np.load(os.path.join(CLUSTERING_PATH, "ucc_clusters_names.npy"))
    UCC_clusters_quality_class = np.load(os.path.join(CLUSTERING_PATH, "ucc_clusters_quality_class.npy"))

    # Reorder the quality classes by the sorted names to match RPJE
    reorder = np.argsort(UCC_clusters_names)
    UCC_clusters_names = UCC_clusters_names[reorder]
    UCC_clusters_quality_class = UCC_clusters_quality_class[reorder]

    # Weight the recovery, purity, and Jaccard index by the evidence
    print("... weighting the recovery, purity, and Jaccard index by the evidence")
    #RPJE[..., 3] **= 2
    RPJE[..., 0] *= RPJE[..., 3]  # Evidence-weighted recovery
    RPJE[..., 1] *= RPJE[..., 3]  # Evidence-weighted purity
    RPJE[..., 2] *= RPJE[..., 3]  # Evidence-weighted Jaccard index

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the recovery, purity, and Jaccard index for each significance level
    print("... plotting the recovery, purity, and Jaccard index vs significance level for all clusters")    
    sum_of_evidence_weights = np.sum(RPJE[..., 3], axis=1)
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
            np.sum(RPJE[..., 0], axis=1) / sum_of_evidence_weights,
            color='k', linestyle='dotted', linewidth=1.5,
            label='Recovery (all)')
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
            np.sum(RPJE[..., 1], axis=1) / sum_of_evidence_weights,
            color='k', linestyle='dashed', linewidth=1.5,
            label='Purity (all)')
    ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
            np.sum(RPJE[..., 2], axis=1) / sum_of_evidence_weights,
            color='k', linestyle='solid', linewidth=1.5,
            label='Jaccard index (all)')


    idx = np.argmax(np.sum(RPJE[..., 2], axis=1) / sum_of_evidence_weights)
    print('... best fit S =', SIGMA_THRESHOLDS_FOR_COMPARISONS[idx], 'with', num_astrolink_clusters[idx], 'clusters')

    # Plot the recovery, purity, and Jaccard index for each significance level for each cluster type
    print("... plotting the recovery, purity, and Jaccard index vs significance level for different quality ranges")
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
        mask = np.zeros(RPJE.shape[1], dtype=np.bool_)
        for quality_class in class_list:
            mask[UCC_clusters_quality_class == quality_class] = True
        sum_of_evidence_weights = np.sum(RPJE[:, mask, 3], axis=1)
        ax.plot(SIGMA_THRESHOLDS_FOR_COMPARISONS, 
                np.sum(RPJE[:, mask, 2], axis=1) / sum_of_evidence_weights,
                color=colour, linestyle='solid', linewidth=0.75, alpha=0.75,
                label='Jaccard index (' + ','.join(class_list) + ')')

    print('... saving figure.\n')
    ax.set_xlim(SIGMA_THRESHOLDS_FOR_COMPARISONS.min(), SIGMA_THRESHOLDS_FOR_COMPARISONS.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Significance Level")
    ax.set_ylabel("Comparison Statistic")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close(fig)



# === Run script ===
if __name__ == "__main__":
    # Ensure paths exist
    os.makedirs(REDUCED_CATALOGUE_PATH, exist_ok=True)
    os.makedirs(SUBSAMPLE_PATH, exist_ok=True)
    os.makedirs(CLUSTERING_PATH, exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Reduce raw Gaia catalogue to numpy files
    reduce_gdr3_catalogue_to_numpy_files()
    reduce_bailerjones_gedr3_distances_to_numpy_files()

    # Calculate empirical selection function
    calculate_empirical_survey_selection_function()
    plot_limiting_g_band_magnitude_on_sky()

    # Construct subsample and subsample selection function
    construct_subsample_from_full_catalogue()
    calculate_subsample_selection_function()

    # Calculate total selection function for subsample
    calculate_total_selection_function_for_subsample()
    plot_total_selection_function_for_subsample()
    
    # Construct input data to be passed to AstroLink
    calculate_distance_contraction_for_subsample()
    calculate_contracted_data_and_errors_for_subsample()
    construct_cartesian_coordinates_for_subsample()

    # Apply AstroLink to subsample and plot of cluster properties
    apply_astrolink_to_subsample()
    plot_prominence_model_fit()
    plot_number_of_clusters_vs_significance()
    plot_cluster_labels_on_sky()
    plot_cluster_proper_motions_on_sky()

    # Compare clustering output to Hunt & Reffert (2024)
    prepare_for_Hunt2024_comparison()
    compare_to_Hunt2024()
    plot_evidence_weighted_Hunt2024_comparison_results()

    # Compare to Unified Cluster Catalogue
    prepare_for_UCC_comparison()
    compare_to_UCC()
    plot_evidence_weighted_UCC_comparison_results()