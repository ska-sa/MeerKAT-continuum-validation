#!/usr/bin/env python3

"""Input an radio continuum image and produce a validation report (in html)
in a directory named after the image, which summarises several validation
tests/metrics (e.g. astrometry, flux scale, source counts, etc) and whether
the data passed or failed these tests.

Last updated: 27/06/2018

Usage:
  Radio_continuum_validation.py -h | --help
  Radio_continuum_validation.py [-I --fits=<img>] [-M --main=<main-cat>]
  [-N --noise=<map>] [-C --catalogues=<list>] [-F --filter=<config>]
  [-R --snr=<ratio>] [-v --verbose] [-f --refind] [-r --redo] [-p --peak-flux]
  [-w --write] [-x --no-write] [-m --SEDs=<models>] [-e --SEDfig=<extn>]
  [-t --telescope=<name>] [-d --main-dir=<path>] [-n --ncores=<num>]
  [-b --nbins=<num>] [-s --source=<src>] [-a --aegean-params]
  [-c --correct=<level>]

Required:
  -I --fits=<img>           A FITS continuum image [default: None].
  AND/OR
  -M --main=<main-cat>     Use this catalogue config file (overwrites options -p and -t).
  Default is to run Aegean [default: None].

Options:
  -h --help                 Show this help message.
  -C --catalogues=<list>    A comma-separated list of filepaths to catalogue config files
  corresponding to catalogues to use (will look in --main-dir for each file not found in
  given path) [default: NVSS_config.txt,SUMSS_config.txt,GLEAM_config.txt,TGSS_config.txt].
  -N --noise=<map>          Use this FITS image of the local rms. Default is to run BANE
  [default: None].
  -F --filter=<config>      A config file for filtering the sources in the input FITS file
  [default: None].
  -R --snr=<ratio>          The signal-to-noise ratio cut to apply to the input catalogue
  and the source counts (doesn't affect source finding) [default: 5.0].
  -v --verbose              Verbose output [default: False].
  -f --refind               Force source finding step when catalogue already exists
  (sets redo to True) [default: False].
  -r --redo                 Force every step again (except source finding), even when
  catalogues already exist [default: False].
  -p --peak-flux            Use the peak flux rather than the integrated flux of the input image
  (not used when -A used) [default: False].
  -w --write                Write intermediate files generated during processing (e.g.
  cross-matched and pre-filtered catalogues, etc). This will save having to reprocess the
  cross-matches, etc when executing the script again. [default: False].
  -x --no-write             Don't write any files except the html report and any files output from
  BANE and Aegean. [default: False].
  -m --SEDs=<models>        A comma-separated list of SED models to fit to the radio spectra
  ('pow','SSA','FFA','curve',etc) [default: None].
  -e --SEDfig=<extn>        Write figures for each SED model with this file extension
  (may significantly slow down script) [default: None].
  -t --telescope=<name>     Unique name of the telescope or survey to give to the main catalogue
  (not used when -A used). [default: ASKAP].
  -d --main-dir=<path>      The absolute path to the main directory where this script and other
  required files are located [default: $ACES/UserScripts/col52r].
  -n --ncores=<num>         The number of cores (per node) to use when running BANE and Aegean
  (using >=20 cores may result in memory error) [default: 8].
  -b --nbins=<num>          The number of bins to use when performing source counts [default: 50].
  -s --source=<src>         The format for writing plots (e.g. screen, html, eps, pdf, png, etc)
  [default: html].
  -a --aegean=<params>      A single string with any extra paramters to pass into Aegean
  (except cores, noise, background, and table) [default: --floodclip=3].
  -c --correct=<level>      Correct the input FITS image, write to 'name_corrected.fits'
  and use this to run through 2nd iteration of validation. FITS image is corrected according
  to input level (0: none, 1: positions, 2: positions + fluxes) [default: 0]."""

import glob
import os
import sys
import warnings
from inspect import currentframe, getframeinfo

import matplotlib as mpl
from docopt import docopt

from catalogue import catalogue
from functions import parse_string, new_path, changeDir, find_file, config2dic
from radio_image import radio_image
from report import report

cf = currentframe()
WARN = '\n\033[91mWARNING: \033[0m' + getframeinfo(cf).filename


def process_args():
    """Process args to be in the required formats."""
    try:
        args = docopt(__doc__)
        if args['--main'] == 'None':
            if args['--fits'] == 'None' or args['--noise'] != 'None':
                raise SyntaxError
    except SyntaxError:
        warnings.warn_explicit("""You must pass in a FITS image (option -I) and/or the main catalogue
        (option -M).\n""", UserWarning, WARN, cf.f_lineno)
        warnings.warn_explicit("""When no catalogue is passed in, you cannot input a noise map
        (option -N).\n""", UserWarning, WARN, cf.f_lineno)
        warnings.warn_explicit("""Use option -h to see usage.\n""", UserWarning, WARN, cf.f_lineno)
        sys.exit()

    # Don't use normal display environment unless user wants to view plots on screen
    if args['--source'] != 'screen':
        mpl.use('Agg')

    # Find directory that contains all the necessary files
    main_dir = args['--main-dir']
    if main_dir.startswith('$ACES') and 'ACES' in list(os.environ.keys()):
        ACES = os.environ['ACES']
        main_dir = main_dir.replace('$ACES', ACES)
    if not os.path.exists('{0}/requirements.txt'.format(main_dir)):
        split = sys.argv[0].split('/')
        script_dir = '/'.join(split[:-1])
        print("Looking in '{0}' for necessary files.".format(script_dir))
        if 'Radio_continuum_validation' in split[-1]:
            main_dir = script_dir
        else:
            warnings.warn_explicit("""Can't find necessary files in main directory
            - {0}.\n""".format(main_dir), UserWarning, WARN, cf.f_lineno)

    # Set paramaters passed in by user
    parms = {}
    parms['main_dir'] = main_dir
    parms['img'] = parse_string(args['--fits'])
    parms['verbose'] = args['--verbose']
    parms['source'] = args['--source']
    parms['refind'] = args['--refind']
    parms['redo'] = args['--redo'] or parms['refind']  # Force redo when refind is True
    parms['use_peak'] = args['--peak-flux']
    parms['write_any'] = not args['--no-write']
    parms['write_all'] = args['--write']
    parms['aegean_params'] = args['--aegean']
    parms['scope'] = args['--telescope']
    parms['ncores'] = int(args['--ncores'])
    parms['nbins'] = int(args['--nbins'])
    parms['level'] = int(args['--correct'])
    parms['snr'] = float(args['--snr'])

    if '*' in args['--catalogues']:
        parms['config_files'] = glob.glob(args['--catalogues'])
        print(parms['config_files'])
    else:
        parms['config_files'] = args['--catalogues'].split(',')

    parms['SEDs'] = args['--SEDs'].split(',')
    parms['SEDextn'] = parse_string(args['--SEDfig'])
    if args['--SEDs'] == 'None':
        parms['SEDs'] = 'pow'
        parms['fit_flux'] = False
    else:
        parms['fit_flux'] = True

    # Force write_all=False write_any=False
    if not parms['write_any']:
        parms['write_all'] = False

    # Add '../' to relative paths of these files, since
    # we'll create and move into a directory for output files
    parms['filter_config'] = new_path(parse_string(args['--filter']))
    parms['main_cat_config'] = parse_string(args['--main'])
    parms['noise'] = new_path(parse_string(args['--noise']))

    return parms


def create_suffix(snr, use_peak):
    # Add S/N and peak/int to output directory/file names
    suffix = 'snr{0}_'.format(snr)
    if use_peak:
        suffix += 'peak'
    else:
        suffix += 'int'
    return suffix


def create_cat(suffix, args):
    """Extract sources for a given image using Aegean or read in supplied catalogue."""
    # Load in FITS image
    if args['img'] is not None:
        changeDir(args['img'], suffix, verbose=args['verbose'])
        img = new_path(args['img'])
        image = radio_image(img, verbose=args['verbose'], rms_map=args['noise'])

        # Run BANE if user hasn't input noise map
        if args['noise'] is None:
            image.run_BANE(ncores=args['ncores'], redo=args['refind'])

        # Run Aegean and create main catalogue object from its output
        if args['main_cat_config'] is None:
            image.run_Aegean(ncores=args['ncores'], redo=args['refind'],
                             params=args['aegean_params'], write=args['write_any'])
            cat = catalogue(image.cat_comp, args['scope'], finder='aegean', image=image,
                            verbose=args['verbose'], autoload=False, use_peak=args['use_peak'])
    else:
        changeDir(args['main_cat_config'], suffix, verbose=args['verbose'])
        image = None

    if args['main_cat_config'] is not None:
        args['main_cat_config'] = new_path(args['main_cat_config'])

        # Use input catalogue config file
        if args['verbose']:
            print("Using config file '{0}' for main catalogue.".format(args['main_cat_config']))
        args['main_cat_config'] = find_file(args['main_cat_config'], args['main_dir'],
                                            verbose=args['verbose'])
        main_cat_config_dic = config2dic(args['main_cat_config'], args['main_dir'],
                                         verbose=args['verbose'])
        main_cat_config_dic.update({'image': image, 'verbose': args['verbose'], 'autoload': False})
        cat = catalogue(**main_cat_config_dic)

    return cat, image


def filter_sources(suffix, args, cat, image):
    """Filter out sources according to given criteria."""
    # Filter out sources below input SNR, and set key fields and create report object before
    # filtering catalogue further so source counts can be written for all sources above input SNR
    cat.filter_sources(SNR=args['snr'], redo=args['redo'], write=args['write_any'],
                       verbose=args['verbose'], file_suffix='_snr{0}'.format(args['snr']))
    cat.set_specs(image)
    myReport = report(cat, args['main_dir'], img=image, verbose=args['verbose'],
                      plot_to=args['source'], redo=args['redo'], src_cnt_bins=args['nbins'],
                      write=args['write_any'])

    # Use config file for filtering sources if it exists
    if args['filter_config'] is not None:
        if args['verbose']:
            print("Using config file '{0}' for filtering.".format(args['filter_config']))
        filter_dic = config2dic(args['filter_config'], args['main_dir'], verbose=args['verbose'])
        filter_dic.update({'redo': args['redo'], 'write': args['write_all'],
                          'verbose': args['verbose']})
        cat.filter_sources(**filter_dic)
    else:
        # Only use reliable point sources for comparison
        cat.filter_sources(flux_lim=1e-3, ratio_frac=1.4, reject_blends=True, flags=True,
                           psf_tol=1.5, resid_tol=3, redo=args['redo'], write=args['write_all'],
                           verbose=args['verbose'])
    return myReport


def match_cat(args, cat):
    """Match the input catalogue/image sources to sources in a list of given catalogues"""
    # Process each catalogue object according to list of input catalogue config files.
    # This will cut out a box, cross-match to this instance, and derive the spectral index.
    for config_file in args['config_files']:
        if args['verbose']:
            print("Using config file '{0}' for catalogue.".format(config_file))
        config_file = config_file.strip()  # In case user put a space
        config_file = find_file(config_file, args['main_dir'], verbose=args['verbose'])
        cat.process_config_file(config_file, args['main_dir'], redo=args['redo'],
                                verbose=args['verbose'], write_all=args['write_all'],
                                write_any=args['write_any'])

    # Derive spectral indices using all fluxes except from main catalogue, and derive the
    # flux at this frequency
    if len(cat.cat_list) > 1:
        cat.fit_spectra(redo=args['redo'], models=args['SEDs'], GLEAM_subbands='int',
                        GLEAM_nchans=None, cat_name=None, write=args['write_any'],
                        fit_flux=args['fit_flux'], fig_extn=args['SEDextn'])


def validate_cat(args, cat, image, myReport):
    """Produce validation report for each cross-matched survey."""
    for cat_name in cat.cat_list[1:]:
        # Print "Would validate {0}".format(cat_name)
        if cat.count[cat_name] > 1:
            myReport.validate(cat.name, cat_name, redo=args['redo'])

    # Write validation summary table and close html file
    myReport.write_html_end()

    # Correct image
    flux_factor = 1.0
    if args['level'] == 2:
        flux_factor = myReport.metric_val['Flux Ratio']
    if args['level'] in (1, 2):
        image.correct_img(myReport.metric_val['RA Offset'], myReport.metric_val['DEC Offset'],
                          flux_factor=flux_factor)


def main():
    """Perform QA validation on an input image or catalogue."""
    args = process_args()
    suffix = create_suffix(args['snr'], args['use_peak'])

    cat, image = create_cat(suffix, args)
    myReport = filter_sources(suffix, args, cat, image)
    match_cat(args, cat)
    validate_cat(args, cat, image, myReport)


if __name__ == "__main__":
    main()
