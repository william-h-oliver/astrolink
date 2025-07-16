# Standard imports
import os

# Set number of parallel workers (note this requires the environment variable to exist before running this script)
from numba import set_num_threads
PARALLEL_WORKERS = min(os.cpu_count(), 64)  # Use up to 64 workers or all available CPUs, whichever is smaller
if PARALLEL_WORKERS != -1:
    os.environ["OMP_NUM_THREADS"] = f"{PARALLEL_WORKERS}"
    set_num_threads(PARALLEL_WORKERS)
else:
    os.environ["OMP_NUM_THREADS"] = f"{os.cpu_count()}"
    set_num_threads(os.cpu_count())

# Remaining standard imports
import gc
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
import numpy as np
import pandas as pd
from pykdtree.kdtree import KDTree
from sklearn import get_config
from sklearn.utils import gen_batches

# Astro-specific imports
from astropy.table import Table # Works using v6.1.3, but v7.1.0 seems to try and convert 'null' values to float before using fill_values
from gaiaunlimited.selectionfunctions import m10_to_completeness

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

# AstroLink imports
from astrolink import AstroLink
from astrolink import io


# === Define script configuration ===
# Define paths
RAW_GDR3_CATALOGUE_PATH = "/home/_data/Gaia/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"  # Path to raw gdr3 catalogue files
BAILERJONES_GEDR3_DISTANCES_FILE = "/home/williamoliver_data/gaia_clustering/bailerjones_gedr3_distances/gedr3dist.dump.gz"  # Path to Bailer-Jones GEDR3 distances
REDUCED_CATALOGUE_PATH = "/home/williamoliver_data/gaia_clustering/catalogue_files/"  # Path to numpy files of reduced catalogue
SUBSAMPLE_PATH = "/home/williamoliver_data/gaia_clustering/subsample_files/"  # Path to numpy files of subsample from full catalogue
CLUSTERING_PATH = "/home/williamoliver_data/gaia_clustering/clustering_output/"  # Path to AstroLink output files
FIGURES_PATH = "/home/williamoliver_data/gaia_clustering/figures/"  # Path to figures

# Working memory for k-nearest-neighbour retrieval
WORKING_MEMORY = get_config()["working_memory"] / 2  # Default is 1GB, but can be set to a higher value in sklearn config

# Pipeline constants
KNN_FOR_SELECTION_FUNCTION = 32 # Number of nearest neighbors for selection function calculations
SURVEY_SF_LOWER_LIMIT = 0.99 # Empirical survey selection function lower limit for subsample stars
HEALPIX_LEVEL = 12 # HEALPix level for on-sky plotting
KNN_FOR_ASTROLINK = 10 # Number of nearest neighbors for AstroLink
SIGMA_FOR_ASTROLINK = 4 # Sigma level for AstroLink



# === Reduce GDR3 and Bailer-Jones GEDR3 catalogues to numpy files ===
def reduce_gdr3_catalogue_to_numpy_files(overwrite=False):
    """
    Reduce raw Gaia catalogue CSV files to grouped numpy arrays.
    """
    # Define column groups for reduction
    column_groups = {
        'source_ids': ['source_id'],
        'galactic_coordinates': ['l', 'b'],
        'equitorial_coordinates': ['ra', 'dec'],
        'parallaxes': ['parallax'],
        'proper_motions': ['pmra', 'pmdec'],
        'astrometric_errors': ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error'],
        'astrometric_matched_transits': ['astrometric_matched_transits'],
        'photometry': ['phot_g_mean_mag'],#, 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
    }

    # Skip processing if all merged output files already exist
    all_exist = all(
        os.path.exists(os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}.npy'))
        for group_name in column_groups
    )
    if all_exist and not overwrite:
        print("All GDR3 catalogue reduction numpy files already exist. Skipping reduction.")
        print("Use overwrite=True to force reprocessing.\n")
        return
    print("Reducing raw Gaia catalogue to numpy files...")

    file_paths = sorted(globTrue(os.path.join(RAW_GDR3_CATALOGUE_PATH, 'GaiaSource_*.csv.gz')))
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
        combined = np.concatenate(arrays)

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
        'r_med_photogeo': ['r_med_photogeo'],
        'r_lo_high_photogeo': ['r_lo_photogeo', 'r_hi_photogeo']
    }

    # Skip processing if all merged output files already exist
    all_exist = all(
        os.path.exists(os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}.npy'))
        for group_name in column_groups
    )
    if all_exist and not overwrite:
        print("All reduced Bailer-Jones distance numpy files already exist. Skipping reduction.")
        print("Use overwrite=True to force reprocessing.\n")
        return
    print("Reducing the Bailer-Jones GEDR3 distance dump file to numpy arrays...")
    
    chunksize = 10**6  # Adjust based on your memory
    all_columns = [col for cols in column_groups.values() for col in cols]
    for i, chunk in enumerate(pd.read_csv(BAILERJONES_GEDR3_DISTANCES_FILE, compression="gzip", chunksize=chunksize, usecols=all_columns)):
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
            gdr3_source_ids = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_source_ids.npy")[:, 0]  # (n,)
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
            final_path = os.path.join(REDUCED_CATALOGUE_PATH, f'gdr3_{group_name}.npy')
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
    file_path_m10_stars = f"{REDUCED_CATALOGUE_PATH}/gdr3_m10_stars.npy"
    file_path_sf = f"{REDUCED_CATALOGUE_PATH}/gdr3_empirical_survey_selection_function.npy"
    file_path_m10_healpix = f"{REDUCED_CATALOGUE_PATH}/gdr3_m10_healpix.npy"
    if os.path.exists(file_path_m10_stars) and os.path.exists(file_path_sf) and os.path.exists(file_path_m10_healpix) and not overwrite:
        print(f"Empirical selection function and m10 values for the centre of HEALpix pixels already exist at:")
        print(f"\t{file_path_sf} and")
        print(f"\t{file_path_m10_healpix}")
        print("Use overwrite=True to recompute.\n")
        return
    print("Calculating empirical survey selection function...")

    # Load required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_galactic_coordinates.npy")  # shape (n, 2)
    G_band_magnitudes = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_photometry.npy")[:, 0]           # shape (n,)
    astrometric_matched_transits = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_astrometric_matched_transits.npy")[:, 0]  # shape (n,)

    # Identify stars with valid G magnitude and also stars with less than 11 astrometric matched transits
    print("... identifying valid G-band magnitudes and astrometric matched transits")
    valid_gmag = np.isfinite(G_band_magnitudes)
    valid_for_kNN = np.where(valid_gmag & (astrometric_matched_transits < 11))[0]  # Indices of stars with valid G-band magnitudes and <11 astrometric matched transits
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
    chunk_n_rows = max(min(int(WORKING_MEMORY * (2**20) // 16*KNN_FOR_SELECTION_FUNCTION), n), 1)
    batches = list(gen_batches(n, chunk_n_rows))
    num_batches = len(batches)

    # Compute m10 for each star as median G of neighbors with <11 transits
    for i, sl in enumerate(batches):
        print(f"... computing m10 values for each star -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        _, idx = tree.query(xyz_stars[sl], k=KNN_FOR_SELECTION_FUNCTION, sqr_dists=True)

        # Median G-band magnitude of neighbors
        m10_stars[sl] = np.median(G_band_magnitudes[valid_for_kNN[idx]], axis=1)

        # Delete temporary variables to free memory
        del _, idx
        gc.collect()

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
    print("... converting HEALpix pixel centres to unit 3D Cartesian coordinates")
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
    chunk_n_rows = max(min(int(WORKING_MEMORY * (2**20) // 16*KNN_FOR_SELECTION_FUNCTION), npix), 1)
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
    file_m10_path = os.path.join(FIGURES_PATH, "limiting_g_mag_m10.png")
    file_limiting_g_mag_path = os.path.join(FIGURES_PATH, "limiting_g_mag_mollview.png")
    if os.path.exists(file_limiting_g_mag_path) and not overwrite:
        print(f"Plots already exists at:")
        print(f"\t{file_m10_path} and")
        print(f"\t{file_limiting_g_mag_path}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Plotting M10 map across the sky...")

    # Load m10 values for HEALPix pixels
    m10 = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_m10_healpix.npy")  # (npix,)

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
    
    print(f"... saved mollview plot to {file_limiting_g_mag_path}\n")


# === Construct subsample and subsample selection function ===
def construct_subsample_from_full_catalogue(overwrite=False):
    """
    Create a boolean subsample mask where the empirical survey selection function S_Gaia > SURVEY_SF_LOWER_LIMIT.
    """
    # Check if subsample mask already exists
    mask_path = os.path.join(SUBSAMPLE_PATH, "gdr3_subsample_mask.npy")
    if os.path.exists(mask_path) and not overwrite:
        print(f"Subsample mask already exists at:\n\t{mask_path}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Constructing subsample from full catalogue...")

    # Load selection function and galactic coordinates
    selection_function = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_empirical_survey_selection_function.npy")
    galactic_coords = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_galactic_coordinates.npy")  # (n, 2) in degrees
    parallax = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_parallaxes.npy")[:, 0]  # (n,)
    proper_motions = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_proper_motions.npy")  # (n, 2) in mas/yr

    # Create boolean mask for S_Gaia > threshold and valid astrometric data
    subsample_mask = np.logical_and(
        selection_function > SURVEY_SF_LOWER_LIMIT,
        np.isfinite(parallax),
        np.isfinite(proper_motions).all(axis=1)
    )

    # Save mask
    np.save(mask_path, subsample_mask)
    print(f"... saved subsample mask to {mask_path} (selected {subsample_mask.sum()} stars)\n")

def calculate_subsample_selection_function(overwrite=False):
    """
    Calculate the subsample selection function using kNN-based metric.
    """
    # Check if total selection function already exists
    file_subsample_sf_stars = os.path.join(SUBSAMPLE_PATH, "gdr3_subsample_selection_function.npy")
    if os.path.exists(file_subsample_sf_stars) and not overwrite:
        print(f"Subsample selection function already exists at:\n\t{file_subsample_sf_stars}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Calculating subsample selection function...")

    # Load required arrays
    print("... loading required arrays")
    galactic_coordinates = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_galactic_coordinates.npy")  # shape (n, 2)
    G_band_magnitudes = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_photometry.npy")[:, 0]           # shape (n,)
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_subsample_mask.npy"))  # shape (n,)

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
    chunk_n_rows = max(min(int(WORKING_MEMORY * (2**20) // 16*KNN_FOR_SELECTION_FUNCTION), n), 1)
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
def calculate_total_selection_function_for_subsample(overwrite=False):
    """
    Calculate the total selection function for the subsample.
    """
    # Check if arrays already exists
    file_nsub_stars = os.path.join(SUBSAMPLE_PATH, "gdr3_nsub_stars.npy")
    file_nmw_stars = os.path.join(SUBSAMPLE_PATH, "gdr3_nmw_stars.npy")
    file_total_sf_mean_stars = os.path.join(SUBSAMPLE_PATH, "gdr3_total_selection_function_mean_stars.npy")
    file_total_sf_var_stars = os.path.join(SUBSAMPLE_PATH, "gdr3_total_selection_function_var_stars.npy")
    file_total_sf_mean_healpix = os.path.join(SUBSAMPLE_PATH, "gdr3_total_selection_function_mean_healpix.npy")
    file_total_sf_var_healpix = os.path.join(SUBSAMPLE_PATH, "gdr3_total_selection_function_var_healpix.npy")
    if os.path.exists(file_nsub_stars) and os.path.exists(file_nmw_stars) and os.path.exists(file_total_sf_mean_stars) and os.path.exists(file_total_sf_var_stars) and os.path.exists(file_total_sf_mean_healpix) and os.path.exists(file_total_sf_var_healpix) and not overwrite:
        print(f"Total selection function arrays already exist at:")
        print(f"\t{file_total_sf_mean_stars},")
        print(f"\t{file_total_sf_var_stars},")
        print(f"\t{file_nsub_stars},")
        print(f"\t{file_nmw_stars},")
        print(f"\t{file_total_sf_mean_healpix}, and")
        print(f"\t{file_total_sf_var_healpix}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Calculating total selection function for the subsample...")

    # Load the required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_galactic_coordinates.npy")
    G_band_magnitudes = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_photometry.npy")[:, 0]
    survey_sf = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_empirical_survey_selection_function.npy"))
    subsample_sf = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_subsample_selection_function.npy"))

    # Identify stars with valid G magnitude
    print("... identifying valid G-band magnitudes")
    valid_gmag = np.isfinite(G_band_magnitudes)
    del G_band_magnitudes  # Free memory
    gc.collect()  # Force garbage collection

    # Calculate the inverse of the empirical survey selection function for the subsample
    inverse_survey_sf = 1 / np.sqrt(survey_sf[valid_gmag]**2 + 1 / KNN_FOR_SELECTION_FUNCTION**2)  # Avoids diverging values and stops the total selection function from being unreasonably small
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
    chunk_n_rows = max(min(int(WORKING_MEMORY * (2**20) // 16*KNN_FOR_SELECTION_FUNCTION), n), 1)
    batches = list(gen_batches(n, chunk_n_rows))
    num_batches = len(batches)

    # Initialize total selection function array for stars in the subsample
    print("... initializing total selection function arrays for stars in the subsample")
    nsub_stars = np.empty(n)
    nmw_stars = np.empty(n)
    total_sf_mean_stars = np.empty(n)
    total_sf_var_stars = np.empty(n)

    # Compute total selection function for each star in the subsample
    for i, sl in enumerate(batches):
        print(f"... computing total selection function for each star in subsample -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        _, idx = tree.query(xyz_stars[sl], k=KNN_FOR_SELECTION_FUNCTION, sqr_dists=True)

        # Total selection function is the posterior distribution Beta(n_sub + 1, n_mw - n_sub + 1)
        nsub_stars_batch = subsample_sf[idx].sum(axis=1)
        nmw_stars_batch = inverse_survey_sf[idx].sum(axis=1)
        nsub_stars[sl] = nsub_stars_batch
        nmw_stars[sl] = nmw_stars_batch
        total_sf_mean_stars[sl] = (nsub_stars_batch + 1) / (nmw_stars_batch + 2)  # Mean of selection function for stars in subsample
        total_sf_var_stars[sl] = (nsub_stars_batch + 1) * (nmw_stars_batch - nsub_stars_batch + 1) / ((nmw_stars_batch + 2)**2 * (nmw_stars_batch + 3))  # Variance of selection function for stars in subsample

        # Delete temporary variables to free memory
        del _, idx, nsub_stars_batch, nmw_stars_batch
        gc.collect()
    print(f"... range of expected number of neighbours in subsample: {nsub_stars.min():.3f} -- {nsub_stars.max():.3f}")
    print(f"... range of expected number of neighbours in Milky Way: {nmw_stars.min():.3f} -- {nmw_stars.max():.3f}")
    print(f"... range of total selection function mean: {total_sf_mean_stars.min():.3f} -- {total_sf_mean_stars.max():.3f}")
    print(f"... range of total selection function variance: {total_sf_var_stars.min():.3f} -- {total_sf_var_stars.max():.3f}")

    # Save total selection function arrays for stars
    print(f"... saving total selection function arrays for stars to:")
    print(f"\t{file_nsub_stars},")
    print(f"\t{file_nmw_stars},")
    print(f"\t{file_total_sf_mean_stars}, and")
    print(f"\t{file_total_sf_var_stars}\n")
    np.save(file_nsub_stars, nsub_stars)
    np.save(file_nmw_stars, nmw_stars)
    np.save(file_total_sf_mean_stars, total_sf_mean_stars)
    np.save(file_total_sf_var_stars, total_sf_var_stars)

    del nsub_stars, nmw_stars, total_sf_mean_stars, total_sf_var_stars, xyz_stars  # Free memory
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
    chunk_n_rows = max(min(int(WORKING_MEMORY * (2**20) // 16*KNN_FOR_SELECTION_FUNCTION), npix), 1)
    batches = list(gen_batches(npix, chunk_n_rows))
    num_batches = len(batches)

    # Initialize arrays for HEALPix pixels
    print("... initializing total selection function arrays for HEALPix pixels")
    total_sf_mean_healpix = np.empty(npix)
    total_sf_var_healpix = np.empty(npix)

    # Compute m10 for each HEALPix pixel as median G of neighbors with <11 transits
    for i, sl in enumerate(batches):
        print(f"... computing total selection function for HEALPix pixels -- batch {i + 1} of {num_batches}")
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
    print(f"\t{file_total_sf_mean_healpix}, and")
    print(f"\t{file_total_sf_var_healpix}.\n")
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
        print(f"Plots already exist at:")
        print(f"\t{file_total_sf_mean_path} and")
        print(f"\t{file_total_sf_var_path}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Plotting total selection function on the sky...")

    # Load total selection function for HEALPix pixels
    total_sf_mean = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_total_selection_function_mean_healpix.npy"))  # (npix,)
    total_sf_var = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_total_selection_function_var_healpix.npy"))  # (npix,)

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
    
    print(f"... saved mollview plot to {file_total_sf_mean_path}\n")

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
    # Placeholder for actual implementation
    print(f"Calculating distance contraction for subsample at {SUBSAMPLE_PATH} using {PARALLEL_WORKERS} workers.\n")
    # Actual code would go here

def plot_distance_contraction_for_subsample(overwrite=False):
    """
    Plot the distance contraction for the subsample.
    """
    # Placeholder for actual implementation
    print(f"Plotting distance contraction for subsample at {SUBSAMPLE_PATH}.\n")
    # Actual code would go here

def compute_contracted_astrometric_representation(overwrite=True):
    """
    Computes f(r), x^, mu, and v^ for a set of stars using 5D astrometric data.
    """
    # Check if the contracted astrometric representation already exists
    file_path_f_r = os.path.join(SUBSAMPLE_PATH, "gdr3_contracted_distance.npy")
    file_path_x_hat = os.path.join(SUBSAMPLE_PATH, "gdr3_unit_position_vector.npy")
    file_path_mu = os.path.join(SUBSAMPLE_PATH, "gdr3_proper_motion_magnitude.npy")
    file_path_v_hat = os.path.join(SUBSAMPLE_PATH, "gdr3_unit_tangential_velocity_vector.npy")
    if os.path.exists(file_path_f_r) and os.path.exists(file_path_x_hat) and os.path.exists(file_path_mu) and os.path.exists(file_path_v_hat) and not overwrite:
        print(f"Contracted astrometric representation already exists at:")
        print(f"\t{file_path_f_r},")
        print(f"\t{file_path_x_hat},")
        print(f"\t{file_path_mu}, and")
        print(f"\t{file_path_v_hat}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Computing contracted astrometric representation for subsample...")

    # Load required arrays
    print("... loading required arrays")
    equitorial_coordinates = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_equitorial_coordinates.npy")
    r = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_r_med_geo.npy") / 1000  # (n,) in kpc
    proper_motions = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_proper_motions.npy")  # (n, 2) in mas/yr
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_subsample_mask.npy"))  # (n,)

    # Convert parallax (mas) to distance (kpc)
    #print("... converting parallaxes to distances")
    #parallax = parallax[subsample_mask]
    #r = np.full_like(parallax, np.inf)  # Initialize with infinity
    #r[parallax > 0] = 1/parallax[parallax > 0]  # Set zero or negative parallaxes to infinity
    #del parallax  # Free memory
    #gc.collect()  # Force garbage collection

    # Contracted distance
    print("... calculating contracted distance f(r)")
    r_half_kpc = 0.5
    f_r = r#r_half_kpc * np.arctan(r[subsample_mask] / r_half_kpc)  # shape (N,)
    del r # Free memory
    gc.collect()  # Force garbage collection

    # Save contracted distance
    print(f"... saving contracted distance to {file_path_f_r} (shape: {f_r.shape})")
    np.save(file_path_f_r, f_r)
    del f_r  # Free memory
    gc.collect()  # Force garbage collection

    # Convert angles to radians
    print("... converting equitorial coordinates to radians")
    ra_rad, dec_rad = np.deg2rad(equitorial_coordinates[subsample_mask]).T  # shape (n, 2) in radians
    del equitorial_coordinates  # Free memory
    gc.collect()  # Force garbage collection

    # Unit position vector x̂
    print("... calculating unit position vector")
    sin_ra = np.sin(ra_rad)
    cos_ra = np.cos(ra_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    x_hat = np.column_stack([
        cos_ra * cos_dec,   # x
        sin_ra * cos_dec,   # y
        sin_dec             # z
    ])  # shape (N, 3)
    del ra_rad, dec_rad  # Free memory
    gc.collect()  # Force garbage collection

    # Save unit position vector
    print(f"... saving unit position vector to {file_path_x_hat} (shape: {x_hat.shape})")
    np.save(file_path_x_hat, x_hat)
    del x_hat  # Free memory
    gc.collect()  # Force garbage collection

    # Proper motion magnitude mu
    print("... calculating proper motion magnitude")
    mu_alpha, mu_delta = proper_motions[subsample_mask].T  # shape (N, 2) in mas/yr
    mu_alpha_star = mu_alpha * cos_dec
    mu = np.sqrt(mu_alpha_star**2 + mu_delta**2)  # shape (N,)
    del proper_motions, mu_alpha  # Free memory
    gc.collect()  # Force garbage collection

    # Save proper motion magnitude
    print(f"... saving proper motion magnitude to {file_path_mu} (shape: {mu.shape})")
    np.save(file_path_mu, mu)

    # Tangential basis vectors
    print("... calculating tangential basis vectors")
    e_alpha = np.column_stack([
        -sin_ra,
         cos_ra,
         np.zeros_like(cos_ra)
    ])  # shape (N, 3)
    e_delta = np.column_stack([
        -cos_ra * sin_dec,
        -sin_ra * sin_dec,
        cos_dec
    ])  # shape (N, 3)
    del sin_ra, cos_ra, sin_dec, cos_dec  # Free memory
    gc.collect()  # Force garbage collection

    # Unit tangential velocity vector v̂
    print("... calculating unit tangential velocity vector")
    v_vec = mu_alpha_star[:, None] * e_alpha + mu_delta[:, None] * e_delta  # shape (N, 3)
    v_hat = np.zeros_like(v_vec)
    valid = mu > 0
    v_hat[valid] = v_vec[valid] / mu[valid, None]  # normalize only where mu > 0

    # Save unit tangential velocity vector
    print(f"... saving unit tangential velocity vector to {file_path_v_hat} (shape: {v_hat.shape}).\n")
    np.save(file_path_v_hat, v_hat)
    del mu_alpha_star, mu_delta, mu, e_alpha, e_delta, v_vec, v_hat  # Free memory
    gc.collect()  # Force garbage collection

def construct_cartesian_coordinates_for_subsample(overwrite=True):
    """
    Calculate the Cartesian-like coordinates for the subsample.
    """
    # Check if Cartesian coordinates already exist
    file_cartesian_coordinates = os.path.join(SUBSAMPLE_PATH, "gdr3_cartesian_coordinates.npy")
    if os.path.exists(file_cartesian_coordinates) and not overwrite:
        print(f"Cartesian coordinates already exist at:\n\t{file_cartesian_coordinates}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Calculating Cartesian coordinates for subsample...")

    # Load required arrays
    print("... loading required arrays")
    f_r = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_contracted_distance.npy"))  # (N,)
    x_hat = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_unit_position_vector.npy"))  # (N, 3)
    mu = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_proper_motion_magnitude.npy"))  # (N,)
    v_hat = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_unit_tangential_velocity_vector.npy"))  # (N, 3)

    # Calculate Cartesian coordinates
    print("... calculating Cartesian coordinates")
    x = f_r[:, None] * x_hat  # shape (N, 3)
    v = f_r[:, None] * (mu[:, None] * v_hat)  # shape (N, 3)
    del f_r, x_hat, mu, v_hat  # Free memory
    gc.collect()  # Force garbage collection
    
    # Balance the Cartesian coordinates
    print("... balancing Cartesian coordinates")
    x /= np.sqrt(np.var(x, axis=0).sum())  # Scale positions
    v /= np.sqrt(np.var(v, axis=0).sum())  # Scale velocities
    cartesian_coordinates = np.concatenate([x, v], axis=1)  # shape (N, 6)

    # Save Cartesian coordinates
    print(f"... saving Cartesian coordinates to {file_cartesian_coordinates} (shape: {cartesian_coordinates.shape}).\n")
    np.save(file_cartesian_coordinates, cartesian_coordinates)


# === Apply AstroLink to subsample ===
def apply_astrolink_to_subsample(overwrite=True):
    """
    Run AstroLink clustering on the subsample.
    """
    # Check if AstroLink clustering output already exists
    file_astrolink_object = os.path.join(CLUSTERING_PATH, "astrolink_object.npz")
    if os.path.exists(file_astrolink_object) and not overwrite:
        print(f"AstroLink clustering output already exists at:\n\t{file_astrolink_object}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Running AstroLink clustering on the subsample...")

    # Load the required arrays
    print("... loading required arrays for AstroLink clustering")
    cartesian_coordinates = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_cartesian_coordinates.npy"))  # (N, 6)
    total_sf_mean = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_total_selection_function_mean_stars.npy"))  # (N,)

    # Reduce total selection function mean to subsample
    G_band_magnitudes = np.load(os.path.join(REDUCED_CATALOGUE_PATH, "gdr3_photometry.npy"))[:, 0]  # (N,)
    valid_gmag = np.isfinite(G_band_magnitudes)  # Identify stars with valid G-band magnitudes
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_subsample_mask.npy"))  # (N,)
    total_sf_mean = total_sf_mean[subsample_mask[valid_gmag]]  # Filter by subsample mask
    del G_band_magnitudes, valid_gmag, subsample_mask  # Free memory
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

def plot_clusters_on_sky(overwrite=True):
    """
    Plot the clustering output from AstroLink.
    """
    # Check if plots already exist
    file_clusters_on_sky_path = os.path.join(FIGURES_PATH, "clusters_on_sky.png")
    if os.path.exists(file_clusters_on_sky_path) and not overwrite:
        print(f"Clusters on sky plot already exists at:\n\t{file_clusters_on_sky_path}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Plotting AstroLink clusters on the sky...")

    # Load the AstroLink clustering output
    print("... loading AstroLink clustering output")
    clusterer = io.loadAstroLinkObject(os.path.join(CLUSTERING_PATH, "astrolink_object.npz"))
    clusterer.S = 4
    clusterer.extract_clusters()
    print(f"... found {len(clusterer.clusters) - 1} clusters in the clustering output")

    # Load the required arrays
    print("... loading required arrays for plotting")
    galactic_coordinates = np.load(f"{REDUCED_CATALOGUE_PATH}/gdr3_galactic_coordinates.npy")  # (N, 2) in degrees
    subsample_mask = np.load(os.path.join(SUBSAMPLE_PATH, "gdr3_subsample_mask.npy"))  # (N,)

    # Reduce coordinates to subsample
    galactic_coordinates = galactic_coordinates[subsample_mask]
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


# === Analyze clustering output with respect to ground truth ===
def compare_clustering_output_to_ground_truth(overwrite=False):
    """
    Compare the clustering output to the ground truth.
    """
    # Placeholder for actual implementation
    print(f"Comparing clustering output from {CLUSTERING_PATH} to ground truth.\n")
    # Actual code would go here

def plot_comparison_results(overwrite=False):
    """
    Plot the results of the comparison between clustering output and ground truth.
    """
    # Placeholder for actual implementation
    print(f"Plotting comparison results from {CLUSTERING_PATH}.\n")
    # Actual code would go here


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
    plot_distance_contraction_for_subsample()
    compute_contracted_astrometric_representation()
    construct_cartesian_coordinates_for_subsample()

    # Run AstroLink clustering on subsample
    apply_astrolink_to_subsample()
    plot_clusters_on_sky()

    # Analyze clustering output with respect to ground truth
    compare_clustering_output_to_ground_truth()
    plot_comparison_results()