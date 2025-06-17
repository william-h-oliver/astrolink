# Standard imports
import os
import gc
from glob import glob
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
import numpy as np
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
from healpy.newvisufunc import projview

# === Reduce raw Gaia catalogue to numpy files grouped by column ===
def _process_single_file(file_path, output_dir, column_groups):
    """Process a single GaiaSource CSV file into group-wise .npy files."""
    # Extract chunk name from filename
    chunk_name = os.path.basename(file_path).replace('GaiaSource_', '').replace('.csv.gz', '')

    # Skip processing if all output files for this chunk already exist
    all_exist = all(
        os.path.exists(os.path.join(output_dir, f'gdr3_{group_name}_{chunk_name}.npy'))
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
        out_path = os.path.join(output_dir, f'gdr3_{group_name}_{chunk_name}.npy')
        np.save(out_path, array)

    del table, columns_data, array  # Free memory
    gc.collect()  # Force garbage collection

    return True

def reduce_catalogue_to_numpy(catalogue_path, reduced_catalogue_path, workers, overwrite=False):
    """
    Reduce raw Gaia catalogue CSV files to grouped numpy arrays.

    Parameters
    ----------
    catalogue_path : str
        Path to the directory containing raw GaiaSource CSV files.
    reduced_catalogue_path : str
        Path to the directory where reduced numpy files will be saved.
    workers : int
        Number of parallel workers to use for processing.
    overwrite : bool
        If True, overwrite existing numpy files. If False, skip if files already exist.
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
        os.path.exists(os.path.join(reduced_catalogue_path, f'gdr3_{group_name}.npy'))
        for group_name in column_groups
    )
    if all_exist and not overwrite:
        print("All catalogue reduction numpy files already exist. Skipping reduction.")
        print("Use overwrite=True to force reprocessing.\n")
        return
    print("Reducing raw Gaia catalogue to numpy files...")

    file_paths = sorted(globTrue(os.path.join(catalogue_path, 'GaiaSource_*.csv.gz')))
    print(f"... found {len(file_paths)} source files.")

    # Parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_single_file, file_path, reduced_catalogue_path, column_groups)
            for file_path in file_paths
        ]
        for future in futures:
            future.result()  # Propagate any errors

    # Merge all intermediate .npy files by group
    for group_name in column_groups.keys():
        group_files = sorted(glob(os.path.join(reduced_catalogue_path, f'gdr3_{group_name}_*.npy')))
        arrays = [np.load(f) for f in group_files]
        combined = np.concatenate(arrays)

        final_path = os.path.join(reduced_catalogue_path, f'gdr3_{group_name}.npy')
        np.save(final_path, combined)
        print(f"... saved combined array: {final_path} (shape: {combined.shape})")

        # Delete intermediates
        for f in group_files:
            os.remove(f)
    print("... reduction complete. All column groups saved as .npy files.\n")


# === Create subsample and total selection function ===
def calculate_empirical_survey_selection_function(reduced_catalogue_path, k, healpix_level, overwrite=False):
    """
    Compute the empirical survey selection function using a kNN-based M10 metric
    and save each as a .npy files aligned with the G-band photometry array.
    
    Parameters
    ----------
    reduced_catalogue_path : str
        Directory path to reduced numpy catalogue.
    k : int
        Number of nearest neighbors to use in M10 computation.
    healpix_level : int
        HEALPix level for on-sky projection.
    overwrite : bool
        If True, overwrite existing selection function file. If False, skip if file exists.
    """
    # Check if selection function already exists
    out_path_m10_stars = f"{reduced_catalogue_path}/gdr3_m10_stars.npy"
    out_path_sf = f"{reduced_catalogue_path}/gdr3_empirical_survey_selection_function.npy"
    out_path_m10_healpix = f"{reduced_catalogue_path}/gdr3_m10_healpix.npy"
    if os.path.exists(out_path_m10_stars) and os.path.exists(out_path_sf) and os.path.exists(out_path_m10_healpix) and not overwrite:
        print(f"Selection function already exists at {out_path_sf} and m10 values at the centre of HEALpix pixels already exists at {out_path_m10_healpix}")
        print("Use overwrite=True to recompute.\n")
        return
    print("Calculating empirical survey selection function...")

    # Load required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(f"{reduced_catalogue_path}/gdr3_galactic_coordinates.npy")  # shape (n, 2)
    G_band_magnitudes = np.load(f"{reduced_catalogue_path}/gdr3_photometry.npy")[:, 0]           # shape (n,)
    astrometric_matched_transits = np.load(f"{reduced_catalogue_path}/gdr3_astrometric_matched_transits.npy")[:, 0]  # shape (n,)

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

    # Build KDTree with only those stars with valid G-band magnitudes and with less than 11 astrometric matched transits
    print("... building kNN tree from the unit 3D Cartesian coordinates of valid stars")
    n = xyz_stars.shape[0]
    m10_stars = np.full_like(G_band_magnitudes, np.nan)  # Initialize m10 values for stars
    tree = KDTree(xyz_stars[valid_for_kNN])

    # Batching for memory efficiency
    working_memory = get_config()["working_memory"] / 2  # Use half of the working memory for this operation
    chunk_n_rows = max(min(int(working_memory * (2**20) // 16*k), n), 1)
    batches = list(gen_batches(n, chunk_n_rows))
    num_batches = len(batches)

    # Compute m10 for each star as median G of neighbors with <11 transits
    for i, sl in enumerate(batches):
        print(f"... computing m10 values for each star -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        _, idx = tree.query(xyz_stars[sl], k=k, sqr_dists=True)

        # Median G-band magnitude of neighbors
        m10_stars[sl] = np.median(G_band_magnitudes[valid_for_kNN[idx]], axis=1)

        # Delete temporary variables to free memory
        del _, idx
        gc.collect()

    # Save m10 values for stars
    print(f"... saving m10 values for stars to {out_path_m10_stars} (shape: {m10_stars.shape})")
    np.save(out_path_m10_stars, m10_stars)
    
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
    print(f"... saving empirical survey selection function to {out_path_sf} (valid: {np.isfinite(selection_function).sum()} stars)\n")
    np.save(out_path_sf, selection_function)
    del selection_function  # Free memory
    gc.collect()  # Force garbage collection

    # Also calculate m10 values at the centre of each HEALPix pixel for plotting
    print("Calculating m10 values for HEALPix pixels...")
    nside = 2**healpix_level
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
    working_memory = get_config()["working_memory"] / 2  # Use half of the working memory for this operation
    chunk_n_rows = max(min(int(working_memory * (2**20) // 16*k), npix), 1)
    batches = list(gen_batches(npix, chunk_n_rows))
    num_batches = len(batches)

    # Initialize m10 array for HEALPix pixels
    print("... initializing m10 array for HEALPix pixels")
    m10_healpix = np.empty(npix)

    # Compute m10 for each HEALPix pixel as median G of neighbors with <11 transits
    for i, sl in enumerate(batches):
        print(f"... computing m10 values for HEALPix pixels -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        _, idx = tree.query(xyz_healpix[sl], k=k, sqr_dists=True)

        # Median G-band magnitude of neighbors
        m10_healpix[sl] = np.median(G_band_magnitudes[valid_for_kNN[idx]], axis=1)

        # Delete temporary variables to free memory
        del _, idx
        gc.collect()
    
    del tree, xyz_healpix, valid_for_kNN  # Free memory
    gc.collect()  # Force garbage collection

    # Save m10 values for HEALPix pixels
    print(f"... saving m10 values for HEALPix pixels to {out_path_m10_healpix} (shape: {m10_healpix.shape})\n")
    np.save(out_path_m10_healpix, m10_healpix)
    del m10_healpix  # Free memory
    gc.collect()  # Force garbage collection

def plot_limiting_g_band_magnitude(reduced_catalogue_path, figures_path, S_Gaia_cut, overwrite=False):
    """
    Plot the limiting G-band magnitude across the sky using HEALPix.

    Parameters
    ----------
    reduced_catalogue_path : str
        Directory containing reduced catalogue .npy files.
    figures_path : str
        Directory to save the mollview plot.
    S_Gaia_cut : float
        Completeness threshold to include stars in the subsample.
    overwrite : bool
        If True, overwrite existing plot. If False, skip if plot already exists.
    """
    # Check if plot already exists
    out_m10_path = os.path.join(figures_path, "limiting_g_mag_m10.png")
    out_limiting_g_mag_path = os.path.join(figures_path, "limiting_g_mag_mollview.png")
    if os.path.exists(out_limiting_g_mag_path) and not overwrite:
        print(f"Plots already exists at {out_m10_path} and {out_limiting_g_mag_path}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Plotting M10 map across the sky...")

    # Load m10 values for HEALPix pixels
    m10 = np.load(f"{reduced_catalogue_path}/gdr3_m10_healpix.npy")  # (npix,)

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
    plt.savefig(out_m10_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    
    print(f"... saved mollview plot to {out_m10_path}\n")


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
    limiting_g_band_magnitude = predictedG0 + predictedInvslope * np.arctanh(2 * (1 - S_Gaia_cut) ** (1 / predictedShape) - 1)

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
    plt.savefig(out_limiting_g_mag_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    
    print(f"... saved mollview plot to {out_limiting_g_mag_path}\n")

def create_subsample_from_full_catalogue(reduced_catalogue_path, subsample_path, S_Gaia_cut, overwrite=False):
    """
    Create a boolean subsample mask where the empirical survey selection function S_Gaia > S_Gaia_cut.
    
    Parameters
    ----------
    reduced_catalogue_path : str
        Directory containing reduced catalogue .npy files.
    subsample_path : str
        Directory to save subsample .npy file.
    S_Gaia_cut : float
        Completeness threshold to include stars in the subsample.
    overwrite : bool
        If True, overwrite existing subsample mask. If False, skip if mask already exists.
    """
    # Check if subsample mask already exists
    mask_path = os.path.join(subsample_path, "gdr3_subsample_mask.npy")
    if os.path.exists(mask_path) and not overwrite:
        print(f"Subsample mask already exists at {mask_path}. Use overwrite=True to recompute.\n")
        return
    print("Creating subsample from full catalogue...")

    # Load selection function and galactic coordinates
    selection_function = np.load(f"{reduced_catalogue_path}/gdr3_empirical_survey_selection_function.npy")
    galactic_coords = np.load(f"{reduced_catalogue_path}/gdr3_galactic_coordinates.npy")  # (n, 2) in degrees
    parallax = np.load(f"{reduced_catalogue_path}/gdr3_parallaxes.npy")[:, 0]  # (n,)
    proper_motions = np.load(f"{reduced_catalogue_path}/gdr3_proper_motions.npy")  # (n, 2) in mas/yr

    # Create boolean mask for S_Gaia > threshold and valid astrometric data
    subsample_mask = np.logical_and(
        selection_function > S_Gaia_cut,
        np.isfinite(parallax),
        np.isfinite(proper_motions).all(axis=1)
    )

    # Save mask
    mask_path = os.path.join(subsample_path, "gdr3_subsample_mask.npy")
    np.save(mask_path, subsample_mask)
    print(f"... saved subsample mask to {mask_path} (selected {subsample_mask.sum()} stars)\n")

def calculate_subsample_selection_function(reduced_catalogue_path, subsample_path, k, overwrite=True):
    """
    Calculate the subsample selection function using kNN-based metric.
    
    Parameters
    ----------
    reduced_catalogue_path : str
        Directory containing reduced catalogue .npy files.
    subsample_path : str
        Path to the subsample numpy files.
    k : int
        Number of nearest neighbors to use in total selection function computation.
    healpix_level : int
        HEALPix level for on-sky projection.
    overwrite : bool
        If True, overwrite existing selection function. If False, skip if already exists.
    """
    # Check if total selection function already exists
    out_subsample_sf_stars = os.path.join(subsample_path, "gdr3_subsample_selection_function.npy")
    if os.path.exists(out_subsample_sf_stars) and not overwrite:
        print(f"Subsample selection function already exists at {out_subsample_sf_stars}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Calculating subsample selection function...")

    # Load required arrays
    print("... loading required arrays")
    galactic_coordinates = np.load(f"{reduced_catalogue_path}/gdr3_galactic_coordinates.npy")  # shape (n, 2)
    G_band_magnitudes = np.load(f"{reduced_catalogue_path}/gdr3_photometry.npy")[:, 0]           # shape (n,)
    subsample_mask = np.load(os.path.join(subsample_path, "gdr3_subsample_mask.npy"))  # shape (n,)

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
    working_memory = get_config()["working_memory"] / 2  # Use half of the working memory for this operation
    chunk_n_rows = max(min(int(working_memory * (2**20) // 16*k), n), 1)
    batches = list(gen_batches(n, chunk_n_rows))
    num_batches = len(batches)

    # Compute subsample selection function for each star as fraction of neighbourhood in subsample
    for i, sl in enumerate(batches):
        print(f"... computing m10 values for each star -- batch {i + 1} of {num_batches}")
        # k-nearest neighbours query
        _, idx = tree.query(comp_stars[sl], k=k, sqr_dists=True)

        # Fraction of neighbours in subsample
        subsample_sf[valid_gmag[sl]] = subsample_mask[valid_gmag[idx]].sum(axis=1) / k

        # Delete temporary variables to free memory
        del _, idx
        gc.collect()
    print(f"... subsample selection function range: {np.nanmin(subsample_sf):.3f} -- {np.nanmax(subsample_sf):.3f}")

    # Save subsample selection function for stars
    print(f"... saving subsample selection function for stars to {out_subsample_sf_stars} (shape: {subsample_sf.shape})\n")
    np.save(out_subsample_sf_stars, subsample_sf)
    del subsample_sf, comp_stars, tree  # Free memory
    gc.collect()  # Force garbage collection

def calculate_total_selection_function_for_subsample(reduced_catalogue_path, subsample_path, k, healpix_level, overwrite=True):
    """
    Calculate the total selection function for the subsample.
    
    Parameters
    ----------
    reduced_catalogue_path : str
        Directory containing reduced catalogue .npy files.
    subsample_path : str
        Path to the subsample numpy files.
    k : int
        Number of nearest neighbors to use in total selection function computation.
    healpix_level : int
        HEALPix level for on-sky projection.
    overwrite : bool
        If True, overwrite existing selection function. If False, skip if already exists.
    """
    # Check if arrays already exists
    out_nsub_stars = os.path.join(subsample_path, "gdr3_nsub_stars.npy")
    out_nmw_stars = os.path.join(subsample_path, "gdr3_nmw_stars.npy")
    out_total_sf_mean_stars = os.path.join(subsample_path, "gdr3_total_selection_function_mean_stars.npy")
    out_total_sf_var_stars = os.path.join(subsample_path, "gdr3_total_selection_function_var_stars.npy")
    out_total_sf_mean_healpix = os.path.join(subsample_path, "gdr3_total_selection_function_mean_healpix.npy")
    out_total_sf_var_healpix = os.path.join(subsample_path, "gdr3_total_selection_function_var_healpix.npy")
    if os.path.exists(out_nsub_stars) and os.path.exists(out_nmw_stars) and os.path.exists(out_total_sf_mean_stars) and os.path.exists(out_total_sf_var_stars) and os.path.exists(out_total_sf_mean_healpix) and os.path.exists(out_total_sf_var_healpix) and not overwrite:
        print(f"Total selection function arrays already exist at:")
        print(f"\t{out_total_sf_mean_stars},")
        print(f"\t{out_total_sf_var_stars},")
        print(f"\t{out_nsub_stars},")
        print(f"\t{out_nmw_stars},")
        print(f"\t{out_total_sf_mean_healpix}, and")
        print(f"\t{out_total_sf_var_healpix}.")
        print("Use overwrite=True to recompute.\n")
    print("Calculating total selection function for the subsample...")

    # Load the required arrays
    print("... loading required arrays from reduced catalogue")
    galactic_coordinates = np.load(f"{reduced_catalogue_path}/gdr3_galactic_coordinates.npy")
    G_band_magnitudes = np.load(f"{reduced_catalogue_path}/gdr3_photometry.npy")[:, 0]
    survey_sf = np.load(os.path.join(reduced_catalogue_path, "gdr3_empirical_survey_selection_function.npy"))
    subsample_sf = np.load(os.path.join(subsample_path, "gdr3_subsample_selection_function.npy"))

    # Identify stars with valid G magnitude
    print("... identifying valid G-band magnitudes")
    valid_gmag = np.isfinite(G_band_magnitudes)
    del G_band_magnitudes  # Free memory
    gc.collect()  # Force garbage collection

    # Change mask to array of indices
    #subsample_mask = subsample_mask[valid_gmag]  # Make mask relative to valid G-band magnitudes
    #subsample_indices = np.where(subsample_mask)[0]  # Indices of stars in the subsample (relative to valid G-band magnitudes)

    # Calculate the inverse of the empirical survey selection function for the subsample
    #inverse_survey_sf = 1 / np.sqrt(survey_sf[valid_gmag]**2 + 1 / k**2)  # Avoids diverging values and stops the total selection function from being unreasonably small
    inverse_survey_sf = 1 / survey_sf[valid_gmag]
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
    working_memory = get_config()["working_memory"] / 2  # Use half of the working memory for this operation
    chunk_n_rows = max(min(int(working_memory * (2**20) // 16*k), n), 1)
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
        _, idx = tree.query(xyz_stars[sl], k=k, sqr_dists=True)

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
    print(f"\t{out_nsub_stars},")
    print(f"\t{out_nmw_stars},")
    print(f"\t{out_total_sf_mean_stars}, and")
    print(f"\t{out_total_sf_var_stars}\n")
    np.save(out_nsub_stars, nsub_stars)
    np.save(out_nmw_stars, nmw_stars)
    np.save(out_total_sf_mean_stars, total_sf_mean_stars)
    np.save(out_total_sf_var_stars, total_sf_var_stars)

    del nsub_stars, nmw_stars, total_sf_mean_stars, total_sf_var_stars, xyz_stars  # Free memory
    gc.collect()  # Force garbage collection

    # Also calculate the total selection function values at the centre of each HEALPix pixel for plotting
    print("Calculating total selection function for HEALPix pixels...")
    nside = 2**healpix_level
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
    working_memory = get_config()["working_memory"] / 2  # Use half of the working memory for this operation
    chunk_n_rows = max(min(int(working_memory * (2**20) // 16*k), npix), 1)
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
        _, idx = tree.query(xyz_healpix[sl], k=k, sqr_dists=True)

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
    print(f"\t{out_total_sf_mean_healpix}, and")
    print(f"\t{out_total_sf_var_healpix}\n")
    np.save(out_total_sf_mean_healpix, total_sf_mean_healpix)
    np.save(out_total_sf_var_healpix, total_sf_var_healpix)
    del total_sf_mean_healpix, total_sf_var_healpix  # Free memory
    gc.collect()  # Force garbage collection

def plot_total_selection_function_for_subsample(subsample_path, figures_path, overwrite=True):
    """
    Plot the limiting G-band magnitude across the sky using HEALPix.

    Parameters
    ----------
    subsample_path : str
        Path to the subsample numpy files.
    figures_path : str
        Directory to save the mollview plot.
    overwrite : bool
        If True, overwrite existing plot. If False, skip if plot already exists.
    """
    # Check if plots already exists
    out_total_sf_mean_path = os.path.join(figures_path, "total_selection_function_mean.png")
    out_total_sf_var_path = os.path.join(figures_path, "total_selection_function_var.png")
    if os.path.exists(out_total_sf_mean_path) and os.path.exists(out_total_sf_var_path) and not overwrite:
        print(f"Plots already exist at {out_total_sf_mean_path} and {out_total_sf_var_path}.")
        print("Use overwrite=True to recompute.\n")
        return
    print("Plotting total selection function on the sky...")

    # Load total selection function for HEALPix pixels
    total_sf_mean = np.load(os.path.join(subsample_path, "gdr3_total_selection_function_mean_healpix.npy"))  # (npix,)
    total_sf_var = np.load(os.path.join(subsample_path, "gdr3_total_selection_function_var_healpix.npy"))  # (npix,)

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
    plt.savefig(out_total_sf_mean_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    
    print(f"... saved mollview plot to {out_total_sf_mean_path}\n")

    # Create a Mollweide projection plot of the total selection function variance
    plt.figure(figsize=(12, 6))
    projview(
        total_sf_var,
        coord=["G"],
        nest=True,
        unit=r"Total selection function variance, $Var[S_{\mathrm{total}}]$",
        cb_orientation="horizontal",
        #min=0,
        #max=1,
        cmap="magma",
        projection_type="mollweide",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(out_total_sf_var_path, dpi=300)
    plt.close()
    gc.collect()  # Free memory
    
    print(f"... saved mollview plot to {out_total_sf_var_path}\n")


# === Construct input data to be passed to AstroLink ===
def calculate_distance_contraction_for_subsample(subsample_path, workers, overwrite=False):
    """
    Calculate the distance contraction for the subsample.
    
    Parameters
    ----------
    subsample_path : str
        Path to the subsample numpy files.
    figures_path : str
        Path to save figures related to distance contraction.
    workers : int
        Number of parallel workers to use for processing.
    overwrite : bool
        If True, overwrite existing distance contraction results. If False, skip if results already exist.
    """
    # Placeholder for actual implementation
    print(f"Calculating distance contraction for subsample at {subsample_path} using {workers} workers.")
    # Actual code would go here

def plot_distance_contraction_for_subsample(subsample_path, figures_path, overwrite=False):
    """
    Plot the distance contraction for the subsample.
    
    Parameters
    ----------
    subsample_path : str
        Path to the subsample numpy files.
    figures_path : str
        Path to save figures related to distance contraction.
    overwrite : bool
        If True, overwrite existing plot. If False, skip if plot already exists.
    """
    # Placeholder for actual implementation
    print(f"Plotting distance contraction for subsample at {subsample_path}.")
    # Actual code would go here

def calculate_cartesian_coordinates_for_subsample(subsample_path, workers, overwrite=False):
    """
    Calculate Cartesian-like coordinates for the subsample.
    
    Parameters
    ----------
    subsample_path : str
        Path to the subsample numpy files.
    figures_path : str
        Path to save figures related to Cartesian coordinates.
    workers : int
        Number of parallel workers to use for processing.
    overwrite : bool
        If True, overwrite existing Cartesian coordinates. If False, skip if already exists.
    """
    # Placeholder for actual implementation
    print(f"Calculating Cartesian-like coordinates for subsample at {subsample_path} using {workers} workers.")
    # Actual code would go here


# === Run AstroLink on subsample and have a first look at the clustering ===
def run_astrolink_on_subsample(subsample_path, clustering_output_path, workers, overwrite=False):
    """
    Run AstroLink clustering on the subsample.
    
    Parameters
    ----------

    subsample_path : str
        Path to the subsample numpy files.
    clustering_output_path : str
        Path to save AstroLink output files.
    figures_path : str
        Path to save figures related to clustering.
    workers : int
        Number of parallel workers to use for clustering.
    overwrite : bool
        If True, overwrite existing clustering output. If False, skip if output already exists.
    """
    # Placeholder for actual implementation
    print(f"Running AstroLink on subsample from {subsample_path} to {clustering_output_path} using {workers} workers.")
    # Actual code would go here

def plot_clustering_output(clustering_output_path, figures_path, overwrite=False):
    """
    Plot the clustering output from AstroLink.
    
    Parameters
    ----------
    clustering_output_path : str
        Path to the AstroLink output files.
    figures_path : str
        Path to save figures related to clustering output.
    overwrite : bool
        If True, overwrite existing plots. If False, skip if plots already exist.
    """
    # Placeholder for actual implementation
    print(f"Plotting clustering output from {clustering_output_path}.")
    # Actual code would go here


# === Analyze clustering output with respect to ground truth ===
def compare_clustering_output_to_ground_truth(clustering_output_path, overwrite=False):
    """
    Compare the clustering output to the ground truth.
    
    Parameters
    ----------
    clustering_output_path : str
        Path to the AstroLink output files.
    figures_path : str
        Path to save figures related to the comparison.
    overwrite : bool
        If True, overwrite existing comparison results. If False, skip if results already exist.
    """
    # Placeholder for actual implementation
    print(f"Comparing clustering output from {clustering_output_path} to ground truth.")
    # Actual code would go here

def plot_comparison_results(clustering_output_path, figures_path, overwrite=False):
    """
    Plot the results of the comparison between clustering output and ground truth.
    
    Parameters
    ----------
    clustering_output_path : str
        Path to the AstroLink output files.
    figures_path : str
        Path to save figures related to the comparison results.
    overwrite : bool
        If True, overwrite existing plots. If False, skip if plots already exist.
    """
    # Placeholder for actual implementation
    print(f"Plotting comparison results from {clustering_output_path}.")
    # Actual code would go here


# === Run script ===
if __name__ == "__main__":
    # Define paths
    catalogue_path = "/home/_data/Gaia/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"  # Path to raw gdr3 catalogue files
    reduced_catalogue_path = "/home/williamoliver_data/gaia_clustering/catalogue_files/"  # Path to numpy files of reduced catalogue
    subsample_path = "/home/williamoliver_data/gaia_clustering/sample_files/"  # Path to numpy files of subsample from full catalogue
    clustering_output_path = "/home/williamoliver_data/gaia_clustering/clustering_output/"  # Path to AstroLink output files
    figures_path = "/home/williamoliver_data/gaia_clustering/figures/"  # Path to figures

    # Ensure paths exist
    os.makedirs(reduced_catalogue_path, exist_ok=True)
    os.makedirs(subsample_path, exist_ok=True)
    os.makedirs(clustering_output_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Number of parallel workers
    workers = min(os.cpu_count(), 64)  # Use up to 32 workers or all available CPUs, whichever is smaller
    os.environ["OMP_NUM_THREADS"] = f"{min(workers, os.cpu_count())}" if workers != -1 else f"{os.cpu_count()}" # Note this requires the environment variable to exist before running this script

    # Pipeline constants
    kNN_for_selection_function = 32 # Number of nearest neighbors for M10 and selection function calculations
    S_Gaia_cut = 0.99 # Empirical survey selection function lower limit for subsample stars
    healpix_level_for_sky_plots = 12 # HEALPix level for plotting limiting G-band magnitude

    # Reduce raw catalogue to numpy files
    reduce_catalogue_to_numpy(
        catalogue_path=catalogue_path,
        reduced_catalogue_path=reduced_catalogue_path,
        workers=workers
    )

    # Calculate the empirical survey selection function for all sources
    calculate_empirical_survey_selection_function(
        reduced_catalogue_path=reduced_catalogue_path,
        k=kNN_for_selection_function,
        healpix_level=healpix_level_for_sky_plots,
    )

    # Plot the limiting G-band magnitude as a function of sky position
    plot_limiting_g_band_magnitude(
        reduced_catalogue_path=reduced_catalogue_path,
        figures_path=figures_path,
        S_Gaia_cut=S_Gaia_cut
    )

    # Create subsample from full catalogue using a cut of the empirical survey selection function
    create_subsample_from_full_catalogue(
        reduced_catalogue_path=reduced_catalogue_path,
        subsample_path=subsample_path,
        S_Gaia_cut=S_Gaia_cut
    )

    # Calculate the subsample selection function using kNN-based metric
    calculate_subsample_selection_function(
        reduced_catalogue_path=reduced_catalogue_path,
        subsample_path=subsample_path,
        k=kNN_for_selection_function
    )

    # Calculate total selection function for subsample
    calculate_total_selection_function_for_subsample(
        reduced_catalogue_path=reduced_catalogue_path,
        subsample_path=subsample_path,
        k=kNN_for_selection_function,
        healpix_level=healpix_level_for_sky_plots
    )

    # Plot the total selection function for subsample
    plot_total_selection_function_for_subsample(
        subsample_path=subsample_path,
        figures_path=figures_path
    )

    # Calculate distance contraction for subsample
    calculate_distance_contraction_for_subsample(
        subsample_path=subsample_path,
        workers=workers
    )

    # Plot distance contraction for subsample
    plot_distance_contraction_for_subsample(
        subsample_path=subsample_path,
        figures_path=figures_path
    )

    # Calculate Cartesian-like coordinates for subsample
    calculate_cartesian_coordinates_for_subsample(
        subsample_path=subsample_path,
        workers=workers
    )

    # Run AstroLink clustering on subsample
    run_astrolink_on_subsample(
        subsample_path=subsample_path,
        clustering_output_path=clustering_output_path,
        workers=workers
    )

    # Plot clustering output
    plot_clustering_output(
        clustering_output_path=clustering_output_path,
        figures_path=figures_path
    )

    # Compare clustering output to ground truth
    compare_clustering_output_to_ground_truth(
        clustering_output_path=clustering_output_path
    )

    # Plot comparison results
    plot_comparison_results(
        clustering_output_path=clustering_output_path,
        figures_path=figures_path
    )