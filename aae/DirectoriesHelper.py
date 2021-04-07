from pathlib import Path


def open_dirs(output_dir_name):
    PROJECT_ROOT = Path.cwd()
    output_dir = PROJECT_ROOT / output_dir_name
    output_dir.mkdir(exist_ok=True)

    experiment_dir = output_dir
    experiment_dir.mkdir(exist_ok=True)

    latent_space_dir = experiment_dir / 'latent_space'
    latent_space_dir.mkdir(exist_ok=True)

    reconstruction_dir = experiment_dir / 'reconstruction'
    reconstruction_dir.mkdir(exist_ok=True)

    sampling_dir = experiment_dir / 'sampling'
    sampling_dir.mkdir(exist_ok=True)

    return latent_space_dir, reconstruction_dir, sampling_dir, experiment_dir