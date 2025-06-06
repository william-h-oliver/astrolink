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

    print(f"[PROCESS] {chunk_name} — starting in PID {os.getpid():<15}", end='\r')
    
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

def reduce_catalogue_to_numpy(catalogue_path, reduced_catalogue_path, workers=32, overwrite=False):
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
        print("All output files already exist. Skipping reduction. Use overwrite=True to force reprocessing.\n")
        return
    print("Reducing raw Gaia catalogue to numpy files...")

    file_paths = sorted(globTrue(os.path.join(catalogue_path, 'GaiaSource_*.csv.gz')))
    print(f"Found {len(file_paths)} source files.")

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
        print(f"Saved combined array: {final_path} (shape: {combined.shape})")

        # Delete intermediates
        for f in group_files:
            os.remove(f)
    print("Reduction complete. All column groups saved as .npy files.\n")


# === Calculate empirical survey selection function for all sources ===
def calculate_empirical_survey_selection_function(reduced_catalogue_path, k=32, overwrite=False):
    """
    Compute the empirical survey selection function using a kNN-based M10 metric
    and save it as a .npy file aligned with the G-band photometry array.
    
    Parameters
    ----------
    reduced_catalogue_path : str
        Directory path to reduced numpy catalogue.
    k : int
        Number of nearest neighbors to use in M10 computation.
    overwrite : bool
        If True, overwrite existing selection function file. If False, skip if file exists.
    """
    # Check if selection function already exists
    out_path = f"{reduced_catalogue_path}/gdr3_empirical_survey_selection_function.npy"
    if os.path.exists(out_path) and not overwrite:
        print(f"Selection function already exists at {out_path}. Use overwrite=True to recompute.\n")
        return
    print("Calculating empirical survey selection function...")

    # Load required arrays
    galactic_coordinates = np.load(f"{reduced_catalogue_path}/gdr3_galactic_coordinates.npy")  # shape (n, 2)
    G_band_magnitudes = np.load(f"{reduced_catalogue_path}/gdr3_photometry.npy")[:, 0]           # shape (n,)
    astrometric_matched_transits = np.load(f"{reduced_catalogue_path}/gdr3_astrometric_matched_transits.npy")[:, 0]  # shape (n,)

    # Identify stars with valid G magnitude
    valid_gmag = np.isfinite(G_band_magnitudes)

    # Convert (l, b) in degrees to unit 3D Cartesian coordinates
    l_rad, b_rad = np.deg2rad(galactic_coordinates).T
    xyz = np.column_stack([
        np.cos(b_rad) * np.cos(l_rad),
        np.cos(b_rad) * np.sin(l_rad),
        np.sin(b_rad)
    ])

    # Build KDTree with only those stars with valid G-band magnitudes and with less than 11 astrometric matched transits
    n = valid_gmag.size
    m10 = np.empty(n)
    valid_for_kNN = valid_gmag & (astrometric_matched_transits < 11)
    nbrs = KDTree(xyz[valid_for_kNN])

    # Chunking for memory efficiency
    working_memory = get_config()["working_memory"]
    chunk_n_rows = max(min(int(working_memory * (2**20) // 16*k), n), 1)

    # Compute m10 for each star as median G of neighbors with <11 transits
    for sl in gen_batches(n, chunk_n_rows):
        # k-nearest neighbours query
        _, idx = nbrs.query(xyz[sl], k=k, sqr_dists=True)

        # Median G-band magnitude of neighbors
        m10[sl] = np.median(G_band_magnitudes[idx], axis=1)

    # Compute completeness using m10_to_completeness (only for valid G-band magnitudes)
    selection_function = np.full_like(G_band_magnitudes, np.nan)
    selection_function[valid_gmag] = m10_to_completeness(
        G_band_magnitudes[valid_gmag],
        m10[valid_gmag]
    )

    # Save the result
    out_path = f"{reduced_catalogue_path}/gdr3_empirical_survey_selection_function.npy"
    np.save(out_path, selection_function)
    print(f"Saved selection function to {out_path} (valid: {np.isfinite(selection_function).sum()} stars)")
    print(f"Empirical survey selection function range: {np.nanmin(selection_function):.3f} -- {np.nanmax(selection_function):.3f}\n")


# === Create subsample from full catalogue ===
def create_subsample_from_full_catalogue(reduced_catalogue_path, subsample_path, S_Gaia_cut=0.95, overwrite=False):
    """
    Create a boolean subsample mask where the empirical survey selection function S_Gaia > S_Gaia_cut.
    Also plot the limiting G-band magnitude across the sky using HEALPix.
    
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

    # Create boolean mask for S_Gaia > threshold
    subsample_mask = selection_function > S_Gaia_cut

    # Save mask
    os.makedirs(subsample_path, exist_ok=True)
    mask_path = os.path.join(subsample_path, "gdr3_subsample_mask.npy")
    np.save(mask_path, subsample_mask)
    print(f"Saved subsample mask to {mask_path} (selected {subsample_mask.sum()} stars)\n")


# === Make plot of the limiting G-band magnitude as a function of sky position ===
def plot_limiting_g_band_magnitude(figures_path, S_Gaia_cut=0.95, healpix_level=10, overwrite=False):
    """
    Plot the limiting G-band magnitude across the sky using HEALPix.

    Parameters
    ----------
    figures_path : str
        Directory to save the mollview plot.
    S_Gaia_cut : float
        Completeness threshold to include stars in the subsample.
    healpix_level : int
        HEALPix NSIDE level for sky projection.
    overwrite : bool
        If True, overwrite existing plot. If False, skip if plot already exists.
    """
    # Check if plot already exists
    fig_path = os.path.join(figures_path, "limiting_g_mag_mollview.png")
    if os.path.exists(fig_path) and not overwrite:
        print(f"Plot already exists at {fig_path}. Use overwrite=True to recompute.\n")
        return
    print("Plotting limiting G-band magnitude across the sky...")

    # Set up HEALPix map for plotting (placeholder values)
    

    # Convert (l, b) -> (theta, phi) in radians
    l_deg, b_deg = galactic_coords.T
    theta = np.deg2rad(90 - b_deg)  # colatitude
    phi = np.deg2rad(l_deg)         # longitude
    del galactic_coords, l_deg, b_deg  # Free memory

    # Get healpix indices
    nside = 2**healpix_level_for_plotting
    npix = hp.nside2npix(nside)
    pix_indices = hp.ang2pix(nside, theta, phi)

    # Calculate limiting G-band magnitude (placeholder logic)
    limiting_g_mag = np.full(npix, np.nan)

    # TODO: Fill `limiting_g_mag[pix_indices]` with computed limiting G-band magnitudes

    # Plot with healpy
    plt.figure(figsize=(10, 6))
    hp.mollview(
        limiting_g_mag,
        title="Limiting G-band Magnitude (placeholder)",
        unit="G mag",
        cmap="viridis",
        notext=False
    )
    fig_path = os.path.join(figures_path, "limiting_g_mag_mollview.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Free memory
    del limiting_g_mag, pix_indices, theta, phi
    gc.collect()
    print(f"Saved mollview plot to {fig_path}\n")


# === Function to run AstroLink clustering on subsample ===
def run_astrolink_on_subsample(subsample_path, clustering_output_path, figures_path, workers=32, overwrite=False):
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


# === Function to compare clustering output to ground truth ===
def compare_clustering_output_to_ground_truth(clustering_output_path, figures_path, overwrite=False):
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
    workers = min(os.cpu_count(), 16)

    # Pipeline constants
    kNN_for_m10 = 32 # Number of nearest neighbors for M10 calculation
    S_Gaia_cut = 0.95 # Empirical survey selection function lower limit for subsample stars
    healpix_level_for_plotting = 10 # HEALPix level for plotting limiting G-band magnitude

    # Reduce raw catalogue to numpy files
    reduce_catalogue_to_numpy(
        catalogue_path=catalogue_path,
        reduced_catalogue_path=reduced_catalogue_path,
        workers=workers,
        overwrite=False  # Set to True to force overwrite existing files
    )

    # Calculate the empirical survey selection function for all sources
    calculate_empirical_survey_selection_function(
        reduced_catalogue_path=reduced_catalogue_path,
        k=kNN_for_m10,
        workers=workers,
        overwrite=False  # Set to True to force overwrite existing selection function
    )

    # Create subsample from full catalogue using a cut of the empirical survey selection function
    create_subsample_from_full_catalogue(
        reduced_catalogue_path=reduced_catalogue_path,
        subsample_path=subsample_path,
        S_Gaia_cut=S_Gaia_cut,
        overwrite=False  # Set to True to force overwrite existing subsample mask
    )

    # Plot the limiting G-band magnitude as a function of sky position
    plot_limiting_g_band_magnitude(
        figures_path=figures_path,
        S_Gaia_cut=S_Gaia_cut,
        healpix_level=10,
        overwrite=False  # Set to True to force overwrite existing plot
    )

    # Calculate total selection function for subsample
    calculate_total_selection_function_for_subsample(
        subsample_path=subsample_path,
        figures_path=figures_path,
        workers=workers,
        overwrite=False  # Set to True to force overwrite existing selection function
    )

    # Run AstroLink clustering on subsample
    run_astrolink_on_subsample(
        subsample_path=subsample_path,
        clustering_output_path=clustering_output_path,
        figures_path=figures_path,
        workers=workers,
        overwrite=False  # Set to True to force re-run clustering
    )

    # Compare clustering output to ground truth
    compare_clustering_output_to_ground_truth(
        clustering_output_path=clustering_output_path,
        figures_path=figures_path,
        overwrite=False  # Set to True to force re-run comparison
    )