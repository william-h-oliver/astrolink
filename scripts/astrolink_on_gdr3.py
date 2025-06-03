# Standard imports
import os

# Third-party imports
import numpy as np






# === Run script ===
if __name__ == "__main__":
    # Define paths
    catalogue_path = "/home/_data/Gaia/cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"  # Replace with your GDR3 data file path
    reduced_catalogue_path = "/home/williamoliver_data/gaia_clustering/catalogue_files/"  # Output numpy file path
    sample_path = "/home/williamoliver_data/gaia_clustering/sample_files/"  # Sample file path
    clustering_output_path = "/home/williamoliver_data/gaia_clustering/clustering_output/"  # Clustering output path
    figures_path = "/home/williamoliver_data/gaia_clustering/figures/"  # Figures output path
    
    # Ensure output directories exist and create them if not
    os.makedirs(reduced_catalogue_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(clustering_output_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Number of parallel workers
    workers = min(os.cpu_count(), 32)