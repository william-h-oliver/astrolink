# Standard imports
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
import numpy as np
from astropy.table import Table


# === Reduce raw Gaia catalogue to numpy files grouped by column ===
# ==================================================================
def _process_single_file(file_path, output_dir, column_groups):
    """Process a single GaiaSource CSV file into group-wise .npy files."""
    # Extract chunk name from filename
    chunk_name = os.path.basename(file_path).replace('GaiaSource_', '').replace('.csv.gz', '')

    # Build union of required columns
    all_columns = sorted({col for cols in column_groups.values() for col in cols})

    # Read with astropy
    table = Table.read(file_path, format='ascii.ecsv', include_names=all_columns)

    # Convert and save each group
    for group_name, group_cols in column_groups.items():
        array = table[group_cols].as_array()  # structured array
        out_path = os.path.join(output_dir, f'gdr3_{group_name}_{chunk_name}.npy')
        np.save(out_path, array)

    return True

def reduce_catalogue_to_numpy(catalogue_path, reduced_catalogue_path, workers=32):
    """Reduce raw Gaia catalogue to .npy arrays grouped by column."""
    os.makedirs(reduced_catalogue_path, exist_ok=True)

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

    file_paths = sorted(glob(os.path.join(catalogue_path, 'GaiaSource_*.csv.gz')))
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




# === Create subsample from full catalogue ===
def create_subsample_from_full_catalogue(reduced_catalogue_path, subsample_path, figures_path):
    """
    Create a subsample from the full catalogue for clustering.
    
    Parameters:
    - reduced_catalogue_path: Path to the reduced numpy files.
    - subsample_path: Path to save the subsample files.
    - figures_path: Path to save figures related to the subsample.
    """
    # Placeholder for actual implementation
    print(f"Creating subsample from {reduced_catalogue_path} to {subsample_path}.")
    # Actual code would go here

# === Function to run AstroLink clustering on subsample ===
def run_astrolink_on_subsample(subsample_path, clustering_output_path, figures_path, workers=32):
    """
    Run AstroLink clustering on the subsample.
    
    Parameters:
    - subsample_path: Path to the subsample files.
    - clustering_output_path: Path to save the clustering output files.
    - figures_path: Path to save figures related to the clustering.
    - workers: Number of parallel workers to use.
    """
    # Placeholder for actual implementation
    print(f"Running AstroLink on subsample from {subsample_path} to {clustering_output_path} using {workers} workers.")
    # Actual code would go here

# === Function to compare clustering output to ground truth ===
def compare_clustering_output_to_ground_truth(clustering_output_path, figures_path):
    """
    Compare the clustering output to the ground truth.
    
    Parameters:
    - clustering_output_path: Path to the clustering output files.
    - figures_path: Path to save figures related to the comparison.
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
    
    # Ensure output directories exist and create them if not
    os.makedirs(reduced_catalogue_path, exist_ok=True)
    os.makedirs(subsample_path, exist_ok=True)
    os.makedirs(clustering_output_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Number of parallel workers
    workers = min(os.cpu_count(), 32)

    # Reduce raw catalogue to numpy files
    reduce_catalogue_to_numpy(
        catalogue_path=catalogue_path,
        reduced_catalogue_path=reduced_catalogue_path,
        workers=workers
    )

    # Create subsample from full catalogue
    create_subsample_from_full_catalogue(
        reduced_catalogue_path=reduced_catalogue_path,
        subsample_path=subsample_path,
        figures_path=figures_path
    )

    # Run AstroLink clustering on subsample
    run_astrolink_on subsample(
        subsample_path=subsample_path,
        clustering_output_path=clustering_output_path,
        figures_path=figures_path,
        workers=workers
    )

    # Compare clustering output to ground truth
    compare_clustering_output_to_ground_truth(
        clustering_output_path=clustering_output_path,
        figures_path=figures_path
    )