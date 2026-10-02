# MeerKAT_continuum_validation

This project validates radio continuum images by comparing detected sources against reference catalogues and producing a quality report.

## Installation

Create and activate a Python environment, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The package installs a command-line entry point called:

```bash
continuum-validate --help
```

## Running the validation pipeline

From the repository root, you can run the packaged entry point:

```bash
continuum-validate --fits path/to/image.fits --snr 5 --catalogues NVSS_config.txt,SUMSS_config.txt
```

You can also run the legacy wrapper scripts directly:

```bash
python Radio_continuum_validation.py --fits path/to/image.fits --snr 5
python ASKAP_continuum_validation.py path/to/image.fits
```

## Common options

- `--fits`: input FITS image
- `--main`: or use an existing main catalogue config
- `--catalogues`: comma-separated list of reference catalogues/configs
- `--snr`: source signal-to-noise threshold
- `--noise`: optional RMS map
- `--verbose`: enable verbose logging
- `--write`: write intermediate files
- `--source`: output format for plots (for example `html`, `png`, `pdf`)

Run the help command for the full list of options:

```bash
continuum-validate --help
```

## Output

The pipeline writes validation reports and plots into a directory named from the image being processed, typically under the working directory.
