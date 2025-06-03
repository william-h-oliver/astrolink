# Standard imports
import os

# Third-party imports
import numpy as np






# === Run script ===
if __name__ == "__main__":
    # Define paths
    catalogue_path = "/home/_data/Gaia/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"  # Path to raw gdr3 catalogue files
    reduced_catalogue_path = "/home/williamoliver_data/gaia_clustering/catalogue_files/"  # Path to numpy files of reduced catalogue
    sample_path = "/home/williamoliver_data/gaia_clustering/sample_files/"  # Path to numpy files of subsample from full catalogue
    clustering_output_path = "/home/williamoliver_data/gaia_clustering/clustering_output/"  # Path to AstroLink output files
    figures_path = "/home/williamoliver_data/gaia_clustering/figures/"  # Path to figures
    
    # Ensure output directories exist and create them if not
    os.makedirs(reduced_catalogue_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(clustering_output_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Number of parallel workers
    workers = min(os.cpu_count(), 32)