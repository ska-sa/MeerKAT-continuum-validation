import collections
import os
import warnings
from datetime import datetime
from inspect import currentframe, getframeinfo

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits as f
from astropy.io import votable
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from matplotlib import colors

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors as pc
from plotly.figure_factory._quiver import _Quiver

try:
    from .functions import get_stats, flux_at_freq, get_pixel_area, axis_lim
except ImportError:
    from functions import get_stats, flux_at_freq, get_pixel_area, axis_lim

# ignore annoying astropy warnings and set my own obvious warning output
warnings.simplefilter('ignore', category=AstropyWarning)
cf = currentframe()
WARN = '\n\033[91mWARNING: \033[0m' + getframeinfo(cf).filename


class report(object):

    def __init__(self, cat, main_dir, img=None, plot_to='html', css_style=None,
                 fig_font={'fontname': 'Serif', 'fontsize': 18}, fig_size={'figsize': (8, 8)},
                 label_size={'labelsize': 12}, markers={'s': 20, 'linewidth': 1,
                                                        'marker': 'o', 'color': 'b'},
                 colour_markers={'marker': 'o', 's': 30, 'linewidth': 0}, cmap='plasma',
                 cbins=20, arrows={'color': 'r', 'width': 0.04, 'scale': 20},
                 src_cnt_bins=50, redo=False, write=True, verbose=True, limit_chi=None):

        """Initialise a report object for writing a html report of the image and cross-matches,
        including plots.

        Arguments:
        ----------
        cat : catalogue
            Catalogue object with the data for plotting.
        main_dir : string
            Main directory that contains all the necessary files.

        Keyword arguments:
        ------------------
        img : radio_image
            Radio image object used to write report table. If None, report will not be written,
            but plots will be made.
        plot_to : string
            Where to show or write the plot. Options are:

                'html' - save as a html file using mpld3.

                'screen' - plot to screen [i.e. call plt.show()].

                'extn' - write file with this extension (e.g. 'pdf', 'eps', 'png', etc).

        css_style : string
            A css format to be inserted in <head>.
        fig_font : dict
            Dictionary of kwargs for font name and size for title and axis labels of matplotlib
            figure.
        fig_size : dict
            Dictionary of kwargs to pass into pyplot.figure.
        label_size : dict
            Dictionary of kwargs for tick params.
        markers : dict
            Dictionary of kwargs to pass into pyplot.figure.scatter, etc (when single colour used).
        colour_markers : dict
            Dictionary of kwargs to pass into pyplot.figure.scatter, etc (when colourmap used).
        arrows : dict
            Dictionary of kwargs to pass into pyplot.figure.quiver.
        redo: bool
            Produce all plots and save them, even if the files already exist.
        write : bool
            Write the source counts and figures to file. Input False to only write report.
        verbose : bool
            Verbose output.
        """

        self.cat = cat
        self.img = img
        self.plot_to = plot_to
        self.fig_font = fig_font
        self.fig_size = fig_size
        self.label_size = label_size
        self.markers = markers
        self.colour_markers = colour_markers
        self.arrows = arrows
        self.src_cnt_bins = src_cnt_bins
        self.main_dir = main_dir
        self.redo = redo
        self.write = write
        self.verbose = verbose
        self.limit_chi = limit_chi

        # set name of directory for figures and create if doesn't exist
        self.figDir = 'figures'
        if self.write and not os.path.exists(self.figDir):
            os.mkdir(self.figDir)

        # use css style passed in or default style for CASS web server below
        if css_style is not None:
            self.css_style = css_style
        else:
            self.css_style = """<?php include("base.inc"); ?>
            <meta name="DCTERMS.Creator" lang="en" content="personalName=Collier,Jordan" />
            <meta name="DC.Title" lang="en" content="Continuum Validation Report" />
            <meta name="DC.Description" lang="en" content="Continuum validation report
            summarising science readiness of data via several metrics" />
            <?php standard_head(); ?>
            <style>
                .reportTable {
                    border-collapse: collapse;
                    width: 100%;
                }

                .reportTable th, .reportTable td {
                    padding: 15px;
                    text-align: middle;
                    border-bottom: 1px solid #ddd;
                    vertical-align: top;
                }

                .reportTable tr {
                    text-align:center;
                    vertical-align:middle;
                }

                .reportTable tr:hover{background-color:#f5f5f5}

                #good {
                    background-color:#00FA9A;
                }

                #uncertain {
                    background-color:#FFA500;
                }

                #bad {
                    background-color:#FF6347;
                }

            </style>\n"""
            self.css_style += "<title>{0} Continuum Validation Report</title>\n""".format(
                self.cat.name)

        # filename of html report
        self.name = 'index.html'
        # Open file html file and write css style, title and heading
        self.write_html_head()
        # write table summary of observations and image if radio_image object passed in
        if img is not None:
            self.write_html_img_table(img)
            rms_map = f.open(img.rms_map)[0]
            solid_ang = 0
        # otherwise assume area based on catalogue RA/DEC limits
        else:
            rms_map = None
            solid_ang = self.cat.area*(np.pi/180)**2

        self.write_html_cat_table()

        # plot the int/peak flux as a function of peak flux
        self.int_peak_flux(usePeak=True)

        # write source counts to report using rms map to measure solid angle or approximate
        # solid angle
        if self.cat.name in list(self.cat.flux.keys()):
            self.source_counts(self.cat.flux[self.cat.name], self.cat.freq[self.cat.name],
                               rms_map=rms_map, solid_ang=solid_ang, write=self.write,
                               limit_chi=self.limit_chi)
        else:
            self.sc_red_chi_sq = -1
        # write cross-match table header
        self.write_html_cross_match_table()

        # store dictionary of metrics, where they come from, how many matches they're
        # derived from, and their level (0,1 or 2) spectral index defaults to -99, as
        # there is a likelihood it will not be needed (if Taylor-term imaging is not done)
        # RA and DEC offsets used temporarily and then dropped before final metrics computed
        key_value_pairs = [('Flux Ratio', 0),
                           ('Flux Ratio Uncertainty', 0),
                           ('Positional Offset', 0),
                           ('Positional Offset Uncertainty', 0),
                           ('Resolved Fraction', self.cat.resolved_frac),
                           ('Spectral Index', 0),
                           ('RMS', self.cat.img_rms),
                           ('Source Counts Reduced Chi-squared', self.sc_red_chi_sq),
                           ('RA Offset', 0),
                           ('DEC Offset', 0)]

        self.metric_val = collections.OrderedDict(key_value_pairs)
        self.metric_source = self.metric_val.copy()
        self.metric_count = self.metric_val.copy()
        self.metric_level = self.metric_val.copy()

    def write_html_head(self):

        """Open the report html file and write the head."""

        self.html = open(self.name, 'w')
        self.html.write("""<!DOCTYPE HTML>
        <html lang="en">
        <head>
            {0}
        </head>
        <?php title_bar("atnf"); ?>
        <body>
            <h1 align="middle">{1} Continuum Data Validation Report</h1>""".format(self.css_style,
                                                                                   self.cat.name))

    def write_html_img_table(self, img):

        """Write an observations and image and catalogue report tables derived from FITS image and header.

        Arguments:
        ----------
        img : radio_image
            A radio image object used to write values to the html table."""

        # generate link to confluence page for each project code
        project = img.project
        if project.startswith('AS'):
            project = self.add_html_link("https://confluence.csiro.au/display/askapsst/{0}+"
                                         "Data".format(img.project), img.project, file=False)

        # Write observations report table
        self.html.write("""
        <h2 align="middle">Observations</h2>
        <table class="reportTable">
            <tr>
                <th>SBID</th>
                <th>Project</th>
                <th>Date</th>
                <th>Duration<br>(hours)</th>
                <th>Field Centre</th>
                <th>Central Frequency<br>(MHz)</th>
            </tr>
            <tr>
                    <td>{0}</td>
                    <td>{1}</td>
                    <td>{2}</td>
                    <td>{3}</td>
                    <td>{4}</td>
                    <td>{5:.2f}</td>
                    </tr>
        </table>""".format(img.sbid,
                           project,
                           img.date,
                           img.duration,
                           img.centre,
                           img.freq))

        # Write image report table
        self.html.write("""
        <h2 align="middle">Image</h2>
        <h4 align="middle"><i>File: '{0}'</i></h3>
        <table class="reportTable">
            <tr>
                <th>Synthesised Beam<br>(arcsec)</th>
                <th>Median r.m.s.<br>(uJy)</th>
                <th>Image peak<br>(Jy)</th>
                <th>Dynamic Range</th>
                <th>Sky Area<br>(deg<sup>2</sup>)</th>
            </tr>
            <tr>
                <td>{1:.1f} x {2:.1f}</td>
                <td>{3}</td>
                <td>{4:.2f}</td>
                <td>{5:.0E}</td>
                <td>{6:.2f}</td>
            </tr>
        </table>""".format(img.name,
                           img.bmaj,
                           img.bmin,
                           self.cat.img_rms,
                           self.cat.img_peak,
                           self.cat.dynamic_range,
                           self.cat.area))

    def write_html_cat_table(self):

        """Write an observations and image and catalogue report tables derived from FITS
        image, header and catalogue."""

        flux_type = 'integrated'
        if self.cat.use_peak:
            flux_type = 'peak'
        if self.cat.med_si == -99:
            med_si = ''
        else:
            med_si = '{0:.2f}'.format(self.cat.med_si)

        # Write catalogue report table
        self.html.write("""
        <h2 align="middle">Catalogue</h2>
        <h4 align="middle"><i>File: '{0}'</i></h3>
        <table class="reportTable">
            <tr>
                <th>Source Finder</th>
                <th>Flux Type</th>
                <th>Number of<br>sources (&ge;{1}&sigma;)</th>
                <th>Multi-component<br>islands</th>
                <th>Sum of image flux vs.<br>sum of catalogue flux</th>
                <th>Median in-band spectral index</th>
                <th>Median int/peak flux</th>
                <th>Source Counts<br>&#967;<sub>red</sub><sup>2</sup></th>
            </tr>
            <tr>
                <td>{2}</td>
                <td>{3}</td>
                <td>{4}</td>
                <td>{5}</td>
                <td>{6:.1f} Jy vs. {7:.1f} Jy</td>
                <td>{8}</td>""".format(self.cat.filename,
                                       self.cat.SNR,
                                       self.cat.finder,
                                       flux_type,
                                       self.cat.initial_count,
                                       self.cat.blends,
                                       self.cat.img_flux,
                                       self.cat.cat_flux,
                                       med_si))

    def write_html_cross_match_table(self):

        """Write the header of the cross-matches table."""

        self.html.write("""
        <h2 align="middle">Cross-matches</h2>
        <table class="reportTable">
            <tr>
                <th>Survey</th>
                <th>Frequency<br>(MHz)</th>
                <th>Cross-matches</th>
                <th>Median offset<br>(arcsec)</th>
                <th>Median flux ratio</th>
                <th>Median spectral index</th>
                </tr>""")

    def get_metric_level(self, good_condition, uncertain_condition):

        """Return metric level 1 (good), 2 (uncertain) or 3 (bad), according to the two input conditions.

        Arguments:
        ----------
        good_condition : bool
            Condition for metric being good.
        uncertain_condition : bool
            Condition for metric being uncertain."""

        if good_condition:
            return 1
        if uncertain_condition:
            return 2
        return 3

    def assign_metric_levels(self):

        """Assign level 1 (good), 2 (uncertain) or 3 (bad) to each metric, depending on specific tolerenace
        values. See https://confluence.csiro.au/display/askapsst/Continuum+validation+metrics"""

        for metric in list(self.metric_val.keys()):
            # Remove keys that don't have a valid value (value=-99 or -1111)
            if self.metric_val[metric] == -99 or self.metric_val[metric] == -111:
                self.metric_val.pop(metric)
                self.metric_source.pop(metric)
                self.metric_level.pop(metric)
            else:
                # Flux ratio within 5/10%?
                if metric == 'Flux Ratio':
                    val = np.abs(self.metric_val[metric]-1)
                    good_condition = val < 0.05
                    uncertain_condition = val < 0.1
                    self.metric_source[metric] = 'Median flux density ratio [ASKAP / {0}]'.format(
                        self.metric_source[metric])
                # Uncertainty on flux ratio less than 10/20%?
                elif metric == 'Flux Ratio Uncertainty':
                    good_condition = self.metric_val[metric] < 0.1
                    uncertain_condition = self.metric_val[metric] < 0.2
                    self.metric_source[metric] = 'R.M.S. of median flux density ratio '
                    '[ASKAP / {0}]'.format(self.metric_source[metric])
                    self.metric_source[metric] += ' (estimated from median absolute deviation '
                    'from median)'
                # Positional offset < 1/5 arcsec
                elif metric == 'Positional Offset':
                    good_condition = self.metric_val[metric] < 1
                    uncertain_condition = self.metric_val[metric] < 5
                    self.metric_source[metric] = 'Median positional offset (arcsec) '
                    '[ASKAP-{0}]'.format(self.metric_source[metric])
                # Uncertainty on positional offset < 1/5 arcsec
                elif metric == 'Positional Offset Uncertainty':
                    good_condition = self.metric_val[metric] < 5
                    uncertain_condition = self.metric_val[metric] < 10
                    self.metric_source[metric] = 'R.M.S. of median positional offset (arcsec) '
                    '[ASKAP-{0}]'.format(self.metric_source[metric])
                    self.metric_source[metric] += ' (estimated from median absolute deviation '
                    'from median)'
                # Reduced chi-squared of source counts < 3/50?
                elif metric == 'Source Counts Reduced Chi-squared':
                    good_condition = self.metric_val[metric] < 3
                    uncertain_condition = self.metric_val[metric] < 50
                    self.metric_source[metric] = 'Reduced chi-squared of source counts'
                # Resolved fraction of sources between 5-20%?
                elif metric == 'Resolved Fraction':
                    cond1 = self.metric_val[metric] > 0.05
                    cond2 = self.metric_val[metric] < 0.2
                    good_condition = cond1 and cond2
                    uncertain_condition = self.metric_val[metric] < 0.3
                    self.metric_source[metric] = 'Fraction of sources resolved according to '
                    'int/peak flux densities'
                # Spectral index less than 0.2 away from -0.8?
                elif metric == 'Spectral Index':
                    val = np.abs(self.metric_val[metric]+0.8)
                    good_condition = val < 0.2
                    uncertain_condition = False
                    self.metric_source[metric] = 'Median in-band spectral index'
                elif metric == 'RMS':
                    good_condition = self.metric_val[metric] < 100
                    uncertain_condition = self.metric_val[metric] < 500
                    self.metric_source[metric] = 'Median image R.M.S. (uJy) from noise map'
                # If unknown metric, set it to 3 (bad)
                else:
                    good_condition = False
                    uncertain_condition = False

                # Assign level to metric
                self.metric_level[metric] = self.get_metric_level(good_condition,
                                                                  uncertain_condition)

        if self.img is not None:
            self.write_CASDA_xml()

    def write_pipeline_offset_params(self):

        """Write a txt file with offset params for ASKAPsoft pipeline for user to easily
        import into config file, and then drop them from metrics.
        See http://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/pipelines
        /ScienceFieldContinuumImaging.html?highlight=offset"""

        txt = open('offset_pipeline_params.txt', 'w')
        txt.write("DO_POSITION_OFFSET=true\n")
        txt.write("RA_POSITION_OFFSET={0:.2f}\n".format(-self.metric_val['RA Offset']))
        txt.write("DEC_POSITION_OFFSET={0:.2f}\n".format(-self.metric_val['DEC Offset']))
        txt.close()

        for metric in ['RA Offset', 'DEC Offset']:
            self.metric_val.pop(metric)
            self.metric_source.pop(metric)
            self.metric_level.pop(metric)
            self.metric_count.pop(metric)

    def write_CASDA_xml(self):

        """Write xml table with all metrics for CASDA."""

        tmp_table = Table([list(self.metric_val.keys()), list(self.metric_val.values()),
                           list(self.metric_level.values()), list(self.metric_source.values())],
                          names=['metric_name', 'metric_value', 'metric_status',
                                 'metric_description'],
                          dtype=[str, float, np.int32, str])
        vot = votable.from_table(tmp_table)
        vot.version = 1.3
        table = vot.get_first_table()
        table.params.extend([votable.tree.Param(vot, name="project", datatype="char",
                                                arraysize="*", value=self.img.project)])
        valuefield = table.fields[1]
        valuefield.precision = '2'
        prefix = ''
        if self.img.project != '':
            prefix = '{0}_'.format(self.img.project)
        xml_filename = '{0}CASDA_continuum_validation.xml'.format(prefix)
        votable.writeto(vot, xml_filename)

    def write_html_end(self):

        """Write the end of the html report file (including table of metrics) and close it."""

        # Close cross-matches table and write header of validation summary table
        self.html.write("""
                </td>
            </tr>
        </table>
        <h2 align="middle">{0} continuum validation metrics</h2>
        <table class="reportTable">
            <tr>
                <th>Flux Ratio<br>({0} / {1})</th>
                <th>Flux Ratio Uncertainty<br>({0} / {1})</th>
                <th>Positional Offset (arcsec)<br>({0} &mdash; {2})</th>
                <th>Positional Offset Uncertainty (arcsec)<br>({0} &mdash; {2})</th>
                <th>Resolved Fraction from int/peak Flux<br>({0})</th>
                <th>Source Counts &#967;<sub>red</sub><sup>2</sup><br>({0})</th>
                <th>r.m.s. (uJy)<br>({0})</th>
            """.format(self.cat.name, self.metric_source['Flux Ratio'],
                       self.metric_source['Positional Offset']))

        # Assign levels to each metric
        self.assign_metric_levels()

        # Flag if in-band spectral indices not derived
        spec_index = False
        if 'Spectral Index' in self.metric_val:
            spec_index = True

        if spec_index:
            self.html.write('<th>Median in-band<br>spectral index</th>')

        # Write table with values of metrics and colour them according to level
        self.html.write("""</tr>
        <tr>
            <td {0}>{1:.2f}</td>
            <td {2}>{3:.2f}</td>
            <td {4}>{5:.2f}</td>
            <td {6}>{7:.2f}</td>
            <td {8}>{9:.2f}</td>
            <td {10}>{11:.2f}</td>
            <td {12}>{13}</td>
        """.format(self.html_colour(self.metric_level['Flux Ratio']), self.metric_val['Flux Ratio'],
                        self.html_colour(self.metric_level['Flux Ratio Uncertainty']),
                   self.metric_val['Flux Ratio Uncertainty'],
                        self.html_colour(self.metric_level['Positional Offset']),
                   self.metric_val['Positional Offset'],
                        self.html_colour(self.metric_level['Positional Offset Uncertainty']),
                   self.metric_val['Positional Offset Uncertainty'],
                        self.html_colour(self.metric_level['Resolved Fraction']),
                   self.metric_val['Resolved Fraction'],
                        self.html_colour(self.metric_level['Source Counts Reduced Chi-squared']),
                   self.metric_val['Source Counts Reduced Chi-squared'],
                        self.html_colour(self.metric_level['RMS']), self.metric_val['RMS']))

        if spec_index:
            self.html.write('<td {0}>{1:.2f}</td>'.format(self.html_colour(
                self.metric_level['Spectral Index']), self.metric_val['Spectral Index']))

        by = ''
        if self.cat.name != 'ASKAP':
            by = """ by <a href="mailto:Jordan.Collier@csiro.au">Jordan Collier</a>"""
        # Close table, write time generated, and close html file
        self.html.write("""</tr>
            </table>
                <p><i>Generated at {0}{1}</i></p>
            <?php footer(); ?>
            </body>
        </html>""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), by))
        self.html.close()
        print("Continuum validation report written to '{0}'.".format(self.name))

    def add_html_link(self, target, link, file=True, newline=False):

        """Return the html for a link to a URL or file.

        Arguments:
        ----------
        target : string
            The name of the target (a file or URL).
        link : string
            The link to this file (thumbnail file name or string to list as link name).

        Keyword Arguments:
        ------------------
        file : bool
            The input link is a file (e.g. a thumbnail).
        newline : bool
            Write a newline / html break after the link.

        Returns:
        --------
        html : string
            The html link."""

        html = """<a href="{0}">""".format(target)
        if file:
            html += """<IMG SRC="{0}"></a>""".format(link)
        else:
            html += "{0}</a>".format(link)
        if newline:
            html += "<br>"
        return html

    def text_to_html(self, text):

        """Take a string of text that may include LaTeX, and return the html
        code that will generate it as LaTeX.

        Arguments:
        ----------
        text : string
            A string of text that may include LaTeX.

        Returns:
        --------
        html : string
            The same text readable as html."""

        # This will allow everything between $$ to be generated as LaTeX
        html = """
                    <script type="text/x-mathjax-config">
                      MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
                    </script>
                    <script type="text/javascript"
                      src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
                    </script>
                    <br>

                    """

        # Write a newline / break for each '\n' in string
        for line in text.split('\n'):
            html += line + '<br>'

        return html

    def html_colour(self, level):

        """Return a string representing green, yellow or red in html if level is 1, 2 or 3.

        Arguments:
        ----------
        level : int
            A validation level.

        Returns:
        --------
        colour : string
            The html for green, yellow or red."""

        if level == 1:
            colour = "id='good'"
        elif level == 2:
            colour = "id='uncertain'"
        else:
            colour = "id='bad'"
        return colour

    def int_peak_flux(self, usePeak=False):

        """Plot the int/peak fluxes as a function of peak flux.

        Keyword Arguments:
        ------------------
        usePeak : bool
            Use peak flux as x axis, instead of SNR."""

        ratioCol = '{0}_int_peak_ratio'.format(self.cat.name)
        self.cat.df[ratioCol] = self.cat.df[self.cat.flux_col] / self.cat.df[self.cat.peak_col]
        SNR = self.cat.df[self.cat.peak_col]/self.cat.df[self.cat.rms_val]
        ratio = self.cat.df[ratioCol]
        peak = self.cat.df[self.cat.peak_col]

        xaxis = SNR
        if usePeak:
            xaxis = peak

        # Plot the int/peak flux ratio
        title = "{0} int/peak flux ratio".format(self.cat.name)

        if self.plot_to == 'html':
            if usePeak:
                xlabel = 'Peak flux ({0})'.format(self.cat.flux_unit.replace('j', 'J'))
            else:
                xlabel = 'S/N'
            ylabel = 'Int / Peak Flux Ratio'
        else:
            xlabel = r'$\rm S_{peak}$'
            if usePeak:
                xlabel += ' ({0})'.format(self.cat.flux_unit.replace('j', 'J'))
            else:
                xlabel += r'$\sigma_{rms}}$'
            ylabel = r'${\rm S_{int} / S_{peak}}$'

        if self.plot_to != 'screen':
            filename = '{0}/{1}_int_peak_ratio.{2}'.format(self.figDir, self.cat.name,
                                                           self.plot_to)
        else:
            filename = ''

        # Get non-nan data shared between each used axis as a numpy array
        x, y, c, indices = self.shared_indices(xaxis, yaxis=ratio)

        # Hack to overlay resolved sources in red
        xres, yres = xaxis[self.cat.resolved], ratio[self.cat.resolved]
        resolved_trace = {'x': xres, 'y': yres, 'mode': 'markers', 'color': 'r',
                          'name': 'Resolved', 'size': self.markers['s']}
        leg_labels = ['Unresolved']

        # Derive the statistics of y and store in string
        ymed, ymean, ystd, yerr, ymad = get_stats(ratio)
        if self.plot_to == 'html':
            txt = 'Median Ratio: %.2f\n' % ymed
            txt += 'Mean Ratio: %.2f\n' % ymean
            txt += '\u03C3(Ratio): %.2f\n' % ystd
            txt += '\u03C3(mean Ratio): %.2f' % yerr
        else:
            txt = r'$\widetilde{Ratio}$: %.2f' % ymed + '\n'
            txt += r'$\overline{Ratio}$: %.2f' % ymean + '\n'
            txt += r'$\sigma_{Ratio}$: %.2f' % ystd + '\n'
            txt += r'$\sigma_{\overline{Ratio}}$: %.2f' % yerr

        # Store median int/peak flux ratio and write to report table
        self.int_peak_ratio = ymed
        self.html.write('<td>{0:.2f}<br>'.format(ymed))

        # Plot the int/peak flux ratio
        self.plot(x,
                  y=y,
                  c=c,
                  line_funcs=[self.y1],
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  text=txt,
                  loc='tl',
                  axis_perc=0,
                  filename=filename,
                  leg_labels=leg_labels,
                  redo=self.redo,
                  extra_traces=[resolved_trace],
                  log_x=True, log_y=True)

    def source_counts(self, fluxes, freq, rms_map=None, solid_ang=0, write=True, limit_chi=None):

        """Compute and plot the (differential euclidean) source counts based on the input flux densities.

        Arguments:
        ----------
        fluxes : list-like
            A list of fluxes in Jy.
        freq : float
            The frequency of these fluxes in MHz.

        Keyword arguments:
        ------------------
        rms_map : astropy.io.fits
            A FITS image of the local rms in Jy.
        solid_ang : float
            A fixed solid angle over which the source counts are computed. Only used when
            rms_map is None.
        write : bool
            Write the source counts to file.
        limit_chi : float
            Limit the chi squared calculation to fluxes above this limit."""

        # Derive file names based on user input
        filename = 'screen'
        counts_file = '{0}_source_counts.csv'.format(self.cat.basename)
        if self.plot_to != 'screen':
            filename = '{0}/{1}_source_counts.{2}'.format(self.figDir, self.cat.name, self.plot_to)
        # Read the log of the source counts from Norris+11
        norris_file = None
        try:
            from importlib.resources import files
            pkg_counts = files('continuum_validation.data').joinpath('all_counts.txt')
            if pkg_counts.is_file():
                norris_file = str(pkg_counts)
        except Exception:
            pass
        if norris_file is None:
            candidates = [
                os.path.join(self.main_dir, 'all_counts.txt'),
                os.path.join(self.main_dir, 'src', 'continuum_validation', 'data', 'all_counts.txt'),
                os.path.join(self.main_dir, 'data', 'all_counts.txt'),
            ]
            for c in candidates:
                if os.path.exists(c):
                    norris_file = c
                    break
        if norris_file is None:
            norris_file = '{0}/all_counts.txt'.format(self.main_dir)
        df_Norris = pd.read_table(norris_file, sep=' ')
        x = df_Norris['S']-3  # convert from log of flux in mJy to log of flux in Jy
        y = df_Norris['Counts']
        yerr = (df_Norris['ErrDown'], df_Norris['ErrUp'])

        # Fit 6th degree polynomial to Norris+11 data
        deg = 6
        poly_paras = np.polyfit(x, y, deg)
        f = np.poly1d(poly_paras)
        xlin = np.linspace(min(x)*1.2, max(x)*1.2)
        ylin = f(xlin)

        # Perform source counts if not already written to file or user specifies to re-do
        if not os.path.exists(counts_file) or self.redo:

            # Warn user if they haven't input an rms map or fixed solid angle
            if rms_map is None and solid_ang == 0:
                warnings.warn_explicit("You must input a fixed solid angle or an rms map to"
                                       "compute the source counts!\n", UserWarning, WARN,
                                       cf.f_lineno)
                return

            # Get the number of bins from the user
            nbins = self.src_cnt_bins
            print("Deriving source counts for {0} using {1} bins.".format(self.cat.name, nbins))

            # Normalise the fluxes to 1.4 GHz
            fluxes = flux_at_freq(1400, freq, fluxes, -0.8)

            # Correct for Eddington bias for every flux, assuming Hogg+98 model
            r = self.cat.df[self.cat.flux_col] / self.cat.df[self.cat.rms_val]
            slope = np.polyder(f)
            q = 1.5 - slope(fluxes)
            bias = 0.5 + 0.5*np.sqrt(1 - (4*q+4)/(r**2))

            # q is derived in log space, so correct for the bias in log space
            fluxes = 10**(np.log10(fluxes)/bias)

            if rms_map is not None:
                w = WCS(rms_map.header)
                if self.verbose:
                    print("Using rms map '{0}' to derive solid angle for each flux bin.".format(
                        self.img.rms_map))
                total_area = get_pixel_area(rms_map, flux=100, w=w)[0]
            else:
                total_area = 0

            # Add one more bin and then discard it, since this is dominated by the
            # few brightest sources.
            # We also add one more to the bins since there's one more bin edge
            # than number of bins
            edges = np.percentile(fluxes, np.linspace(0, 100, nbins+2))
            dN, edges = np.histogram(fluxes, bins=edges)
            dN = dN[:-1]
            edges = edges[:-1]

            # Derive the lower and upper edges and dS
            lower = edges[:-1]
            upper = edges[1:]
            dS = upper-lower
            S = np.zeros(len(dN))
            solid_angs = np.zeros(len(dN))

            for i in range(len(dN)):
                # Derive the mean flux from all fluxes in current bin
                indices = (fluxes > lower[i]) & (fluxes < upper[i])
                S[i] = np.mean(fluxes[indices])

                # Get the pixels from the r.m.s. map where SNR*r.m.s. < flux
                if rms_map is not None:
                    solid_angs[i] = get_pixel_area(rms_map, flux=S[i]/self.cat.SNR, w=w)[1]

                # Otherwise use the fixed value passed in
                else:
                    solid_angs[i] = solid_ang

            # Compute the differential Euclidean source counts and uncertanties in linear space
            counts = (S**2.5)*dN/dS/solid_angs
            err = (S**2.5)*np.sqrt(dN)/dS/solid_angs

            # Store these and the log of these values in pandas data frame
            df = pd.DataFrame()
            df['dN'] = dN
            df['area'] = solid_angs/((np.pi/180)**2)
            df['S'] = S
            df['logS'] = np.log10(S)
            df['logCounts'] = np.log10(counts)
            df['logErrUp'] = np.log10(counts+err) - np.log10(counts)
            df['logErrDown'] = np.abs(np.log10(counts-err) - np.log10(counts))

            # Remove all bins with less than 10% of total solid angle
            bad_bins = df['area'] / total_area < 0.1
            output = ['Solid angle for bin S={0:.2f} mJy less than 10% of total image. '
                      'Removing bin.'.format(S) for S in S[np.where(bad_bins)]*1e3]
            if self.verbose:
                for line in output:
                    print(line)
            df = df[~bad_bins]

            if write:
                if self.verbose:
                    print("Writing source counts to '{0}'.".format(counts_file))
                df.to_csv(counts_file, index=False)

        # Otherwise simply read in source counts from file
        else:
            print("File '{0}' already exists. Reading source counts from this file.".format(
                counts_file))
            df = pd.read_csv(counts_file)

        # Create a figure for the source counts
        title = '{0} 1.4 GHz source counts'.format(self.cat.name)  # self.cat.freq[self.cat.name])
        # Write axes using unicode (for html) or LaTeX
        if self.plot_to == 'html':
            ylabel = "log\u2081\u2080 S\u00B2\u22C5\u2075 dN/dS [Jy\u00B9\u22C5\u2075 " \
                "sr\u207B\u00B9]"
            xlabel = "log\u2081\u2080 S [Jy]"
        else:
            ylabel = r"$\log_{10}$ S$^{2.5}$ dN/dS [Jy$^{1.5}$ sr$^{-1}$]"
            xlabel = r"$\log_{10}$ S [Jy]"

        # For html plots, add labels for the bin centre, count and area for every data point
        labels = ['S: {0:.2f} mJy, dN: {1:.0f}, Area: {2:.2f} deg\u00B2'.format(bin, count, area)
                  for bin, count, area in zip(df['S']*1e3, df['dN'], df['area'])]

        # Derive the square of the residuals (chi squared), and their sum
        # Divided by the number of data points (reduced chi squared)
        def calc_red_chi_sq(y, x, yerr):
            chi = ((y-f(x))/yerr)**2
            red_chi_sq = np.sum(chi)/len(y)
            return red_chi_sq

        # Only calculate reduced chi squared to the specified flux limit
        if limit_chi is not None:
            idx = np.where(df['logS'] > np.log10(limit_chi))[0]
            df_idx = df.iloc[idx]
            red_chi_sq = calc_red_chi_sq(df_idx['logCounts'], df_idx['logS'], df_idx['logErrDown'])
        else:
            red_chi_sq = calc_red_chi_sq(df['logCounts'], df['logS'], df['logErrDown'])

        # Store reduced chi squared value
        self.sc_red_chi_sq = red_chi_sq

        # Plot Norris+11 data
        txt = ''
        if self.plot_to == 'html':
            txt += 'Data from <a href="http://adsabs.harvard.edu/abs/2011PASA...28..215N">'\
                'Norris+11</a>'
            txt += ' (updated from <a href="http://adsabs.harvard.edu/abs/2003AJ....125..465H"'\
                '>Hopkins+03</a>)\n'
        # Indicate flux limit in label if one is used
        if limit_chi is not None:
            txt += '\u03C7\u00B2 (reduced): %.2f for S > %.2f \u03BCJy' % (
                red_chi_sq, limit_chi / 1e-6)
        else:
            txt += '\u03C7\u00B2 (reduced): %.2f' % red_chi_sq

        # Legend labels for the Norris data and line, and the ASKAP data
        xlab = 'Norris+11'
        leg_labels = [xlab, '{0}th degree polynomial fit to {1}'.format(deg, xlab), self.cat.name]

        # Write reduced chi squared to report table
        self.html.write('</td><td>{0:.2f}<br>'.format(red_chi_sq))

        # Norris+11 trace to include in plotly output
        norris_traces = [
            {'x': x, 'y': y, 'mode': 'markers', 'color': 'r', 'name': xlab,
             'size': self.markers['s'], 'yerr': yerr},
            {'x': xlin, 'y': ylin, 'mode': 'lines', 'color': 'k', 'dash': 'dash',
             'name': '{0}th degree polynomial fit to {1}'.format(deg, xlab)},
            ]

        # Plot ASKAP data on top of Norris+11 data
        self.plot(df['logS'],
                  y=df['logCounts'],
                  yerr=(df['logErrDown'], df['logErrUp']),
                  title=title,
                  labels=labels,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  axis_perc=0,
                  text=txt,
                  loc='br',
                  leg_labels=leg_labels,
                  filename=filename,
                  redo=self.redo,
                  vline=limit_chi,
                  extra_traces=norris_traces)

        self.html.write("""</td>
                        </tr>
                    </table>""")

    def x(self, x, y):

        """For given x and y data, return a line at y=x.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            The same list of x values.
        y : list-like
            The list of x values."""

        return x, x

    def y0(self, x, y):

        """For given x and y data, return a line at y=0.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            The same list of x values.
        y : list-like
            A list of zeros."""

        return x, x*0

    def y1(self, x, y):

        """For given x and y data, return a line at y=1.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            The same list of x values.
        y : list-like
            A list of ones."""

        return x, [1]*len(x)

    def x0(self, x, y):

        """For given x and y data, return a line at x=0.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            A list of zeros.
        y : list-like
            The same list of y values."""

        return y*0, y

    def ratio_err_max(self, SNR, ratio):

        """For given x and y data (flux ratio as a function of S/N), return
        the maximum uncertainty in flux ratio.

        Arguments:
        ----------
        SNR : list-like
            A list of S/N ratios.
        ratio : list-like
            A list of flux ratios.

        Returns:
        --------
        SNR : list-like
            All S/N values > 0.
        ratio : list-like
            The maximum uncertainty in the flux ratio for S/N values > 0."""

        return SNR[SNR > 0], 1+3*np.sqrt(2)/SNR[SNR > 0]

    def ratio_err_min(self, SNR, ratio):

        """For given x and y data (flux ratio as a function of S/N), return the
        minimum uncertainty in flux ratio.

        Arguments:
        ----------
        SNR : list-like
            A list of S/N ratios.
        ratio : list-like
            A list of flux ratios.

        Returns:
        --------
        SNR : list-like
            All S/N values > 0.
        ratio : list-like
            The minimum uncertainty in the flux ratio for S/N values > 0."""

        return SNR[SNR > 0], 1-3*np.sqrt(2)/SNR[SNR > 0]

    def axis_to_np(self, axis):

        """Return a numpy array of the non-nan data from the input axis.

        Arguments:
        ----------
        axis : string or numpy.array or pandas.Series or list
            The data for a certain axis. String are interpreted as column names from catalogue
            object passed into constructor.

        Returns:
        --------
        axis : numpy.array
            All non-nan values of the data.

        See Also
        --------
        numpy.array
        pandas.Series"""

        # Convert input to numpy array
        if type(axis) is str:
            axis = self.cat.df[axis].values
        elif axis is pd.Series:
            axis = axis.values

        return axis

    def shared_indices(self, xaxis, yaxis=None, caxis=None):

        """Return a list of non-nan indices shared between all used axes.

        Arguments:
        ----------
        xaxis : string or numpy.array or pandas.Series or list
            A list of the x axis data. String are interpreted as column names from catalogue
            object passed into constructor.
        yaxis : string or numpy.array or pandas.Series or list
            A list of the y axis data. String are interpreted as column names from catalogue object
            passed into constructor. If this is None, yaxis and caxis will be ignored.
        caxis : string or numpy.array or pandas.Series or list
            A list of the colour axis data. String are interpreted as column names from catalogue
            object passed into constructor. If this is None, caxis will be ignored.

        Returns:
        --------
        x : list
            The non-nan x data shared between all used axes.
        y : list
            The non-nan y data shared between all used axes. None returned if yaxis is None.
        c : list
            The non-nan colour data shared between all used axes. None returned if yaxis or
            caxis are None.
        indices : list
            The non-nan indices.

        See Also
        --------
        numpy.array
        pandas.Series"""

        # Convert each axis to numpy array (or leave as None)
        x = self.axis_to_np(xaxis)
        y = self.axis_to_np(yaxis)
        c = self.axis_to_np(caxis)

        # Get all shared indices from used axes that aren't nan
        if yaxis is None:
            indices = np.where(~np.isnan(x))[0]
            return x[indices], None, None, indices
        elif caxis is None:
            indices = np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
            return x[indices], y[indices], None, indices
        else:
            indices = np.where((~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(c)))[0]
            return x[indices], y[indices], c[indices], indices

 
    def mpl_to_css_color(self, c):
    
        """Convert a matplotlib-style colour (shorthand, name, hex, RGBA tuple) to a
        hex string plotly accepts.
    
        Arguments:
        ----------
        c : any matplotlib colour spec
            e.g. 'b', 'red', '#0000ff', (0, 0, 1, 1)
    
        Returns:
        --------
        hex_colour : string"""
    
        return colors.to_hex(c, keep_alpha=True)
    
    
    def mpl_size_to_plotly(self, s, scale=1.2):
    
        """Convert a matplotlib marker size (area, points^2) to a plotly marker size
        (diameter, pixels).
    
        Arguments:
        ----------
        s : float
            Matplotlib marker 's' value.
    
        Keyword Arguments:
        ------------------
        scale : float
            Extra scale factor to tune final on-screen size.
    
        Returns:
        --------
        size : float"""
    
        return np.sqrt(s) * scale
    
    
    def _loc_to_paper_xy(self, loc):
    
        """Map a corner code to plotly paper-relative annotation position.
    
        Arguments:
        ----------
        loc : string
            One of 'tl', 'tr', 'bl', 'br'.
    
        Returns:
        --------
        x, y, xanchor, yanchor : float, float, string, string"""
    
        return {
            'tl': (0.02, 0.98, 'left', 'top'),
            'tr': (0.98, 0.98, 'right', 'top'),
            'br': (0.98, 0.02, 'right', 'bottom'),
            'bl': (0.02, 0.02, 'left', 'bottom'),
        }.get(loc, (0.02, 0.02, 'left', 'bottom'))
    
    def build_plotly_figure(self, x, y=None, c=None, yerr=None, clabel='', title='',
                            xlabel='', ylabel='', labels=None, line_funcs=None,
                            ellipses=None, arrows=None, leg_labels='', axis_perc=10,
                            vline=None, extra_traces=None, log_x=False, log_y=False,
                            reverse_x=False, text=None, loc='bl', equal_aspect=False,
                            xlim=None, ylim=None):
    
        """Build the plotly figure for a report plot. This plotting path is used for 
        html, screen and file-extension ('png', 'pdf', etc) output.
    
        Arguments:
        ----------
        x : numpy.array
            Data for the x axis (or the only axis, for a histogram).
    
        Keyword Arguments:
        ------------------
        y : numpy.array
            Data for the y axis. None plots a histogram of x.
        c : numpy.array
            Data for the colour axis. None uses a single fixed colour.
        yerr : tuple of numpy.array
            (down, up) uncertainties on y.
        clabel, title, xlabel, ylabel : string
            Axis/plot labels.
        labels : list
            Per-point hover text.
        line_funcs : list of callables
            Functions of (xlin, ylin) returning a line to overlay (e.g. self.y1).
        ellipses : list of dict
            Each dict: {'center': (cx, cy), 'width', 'height', 'color', 'dash'}.
        arrows : tuple of (dx, dy)
            Offset vectors for a quiver plot at each (x, y).
        leg_labels : list
            Legend entry for the main data trace (last element used).
        axis_perc : float
            Trim axis limits to this percentile beyond the data range. 0 for no trim.
        vline : float
            Draw a vertical dashed line at log10(vline).
        extra_traces : list of dict
            Additional overlay data (resolved-source highlight, comparison survey data,
            polynomial fit line, etc). Each dict: {'x', 'y', 'mode' ('markers'/'lines'),
            'color', 'name', 'dash', 'size', 'yerr'}.
        log_x, log_y : bool
            Log-scale the respective axis.
        reverse_x : bool
            Reverse the x axis (used for RA vs Dec sky plots).
        text : string
            Annotate this text (stats block) onto the figure.
        loc : string
            Corner for the text annotation: 'tl', 'tr', 'bl', 'br'.
    
        Returns:
        --------
        fig : plotly.graph_objects.Figure"""
    
        fig = go.Figure()
    
        # Histogram (y is None) — spectral index plot
        if y is None:
            edges = np.linspace(-3, 2, 11)
            fig.add_trace(go.Histogram(
                x=x,
                xbins=dict(start=edges[0], end=edges[-1], size=edges[1] - edges[0]),
                marker=dict(color=self.mpl_to_css_color(self.markers['color']),
                        line=dict(color='black', width=0.5)),
                hovertemplate="%{x}<br>N: %{y}<extra></extra>",
            ))
    
        # Scatter, single colour
        elif c is None:
            hover = "x: %{x:.3f}<br>y: %{y:.3f}"
            if labels is not None:
                hover += "<br>%{text}"
            error_y = None
            if yerr is not None:
                down, up = yerr
                error_y = dict(type='data', array=np.asarray(up), arrayminus=np.asarray(down),
                            visible=True, color=self.mpl_to_css_color(self.markers['color']))
            fig.add_trace(go.Scattergl(
                x=x, y=y, mode='markers',
                marker=dict(color=self.mpl_to_css_color(self.markers['color']),
                        size=self.mpl_size_to_plotly(self.markers['s'])),
                error_y=error_y,
                text=labels,
                name=leg_labels[-1] if leg_labels else None,
                showlegend=bool(leg_labels),
                hovertemplate=hover + "<extra></extra>",
            ))
    
        # Scatter, colour axis (S/N, flux ratio, etc.)
        else:
            hover = "x: %{x:.3f}<br>y: %{y:.3f}<br>" + clabel + ": %{marker.color:.3f}"
            if labels is not None:
                hover += "<br>%{text}"
            fig.add_trace(go.Scattergl(
                x=x, y=y, mode='markers',
                marker=dict(color=c, colorscale='Plasma', showscale=True,
                        colorbar=dict(title=clabel),
                        size=self.mpl_size_to_plotly(self.colour_markers['s'])),
                text=labels,
                hovertemplate=hover + "<extra></extra>",
            ))
    
        # Extra overlay data (resolved-source highlight, comparison survey data,
        # polynomial fit line)
        if extra_traces is not None:
            for tr in extra_traces:
                mode = tr.get('mode', 'markers')
                colour = self.mpl_to_css_color(tr.get('color', 'k'))
                if mode == 'markers':
                    error_y = None
                    if tr.get('yerr') is not None:
                        down, up = tr['yerr']
                        error_y = dict(type='data', array=np.asarray(up),
                                    arrayminus=np.asarray(down), visible=True, color=colour)
                    fig.add_trace(go.Scattergl(
                        x=tr['x'], y=tr['y'], mode='markers',
                        marker=dict(color=colour,
                                size=self.mpl_size_to_plotly(tr.get('size', self.markers['s']))),
                        error_y=error_y,
                        name=tr.get('name'),
                        showlegend=tr.get('name') is not None,
                        hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=tr['x'], y=tr['y'], mode='lines',
                        line=dict(color=colour, width=2, dash=tr.get('dash', 'solid')),
                        name=tr.get('name'),
                        showlegend=tr.get('name') is not None,
                        hoverinfo='skip',
                    ))
    
        # Reference lines (y=x, y=0, y=1, x=0, error envelopes, etc.)
        if line_funcs is not None:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            ymin = np.nanmin(y) if y is not None else 0
            ymax = np.nanmax(y) if y is not None else 1
            xlin = np.linspace(xmin, xmax, num=200)
            ylin = np.linspace(ymin, ymax, num=200)
            for func in line_funcs:
                xline, yline = func(xlin, ylin)
                fig.add_trace(go.Scatter(x=xline, y=yline, mode='lines',
                                        line=dict(color='black', width=2),
                                        showlegend=False, hoverinfo='skip'))
    
        # Ellipses (astrometry offset: median position + search radius)
        if ellipses is not None:
            for e in ellipses:
                cx, cy = e['center']
                w, h = e['width'], e['height']
                # Guard against NaN/Inf/zero dimensions (e.g. std of a near-empty
                # cross-match set) 
                if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(w)
                        and np.isfinite(h)) or w <= 0 or h <= 0:
                    if self.verbose:
                        print("Skipping ellipse with invalid dimensions "
                            "(centre={0}, width={1}, height={2}).".format((cx, cy), w, h))
                    continue
                fig.add_shape(type='circle', xref='x', yref='y',
                            x0=cx - w / 2, y0=cy - h / 2, x1=cx + w / 2, y1=cy + h / 2,
                            line=dict(color=self.mpl_to_css_color(e.get('color', 'k')),
                                    dash=e.get('dash', 'solid'), width=2))
    
        # Arrows (positional offsets by sky position)
        if arrows is not None:
            dx, dy = np.asarray(arrows[0]), np.asarray(arrows[1])
            arrow_scale = self.arrows.get('arrow_scale', 0.25)
            arrow_width = self.arrows.get('width', 1.5)
            if arrow_width < 0.5:
                arrow_width = 1.5

            scale = self.arrows.get('scale', 20)
            if scale >= 1:
                scale = 1.0 / scale

            quiver_obj = _Quiver(x, y, dx, dy, scale=scale, arrow_scale=arrow_scale, angle=np.pi / 9)
            bx, by = quiver_obj.get_barbs()
            ax, ay = quiver_obj.get_quiver_arrows()

            # If colour axis is provided, colour arrows to match the scatter plot's colorscale
            if c is not None:
                nbins = 20
                cmin, cmax = np.nanmin(c), np.nanmax(c)
                if cmax > cmin:
                    norm_c = (c - cmin) / (cmax - cmin)
                else:
                    norm_c = np.zeros_like(c)
                norm_c = np.nan_to_num(norm_c, nan=0.0)
                bin_indices = np.clip(np.floor(norm_c * nbins).astype(int), 0, nbins - 1)
                bin_colors = pc.sample_colorscale('Plasma', [(i + 0.5) / nbins for i in range(nbins)])

                for b in range(nbins):
                    idx = np.where(bin_indices == b)[0]
                    if len(idx) == 0:
                        continue
                    seg_x, seg_y = [], []
                    for i in idx:
                        seg_x.extend(bx[3 * i:3 * i + 3] + ax[4 * i:4 * i + 4])
                        seg_y.extend(by[3 * i:3 * i + 3] + ay[4 * i:4 * i + 4])
                    fig.add_trace(go.Scatter(
                        x=seg_x, y=seg_y, mode='lines',
                        line=dict(color=bin_colors[b], width=arrow_width),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            else:
                arrow_colour = self.mpl_to_css_color(self.arrows.get('color', 'r'))
                seg_x = bx + ax
                seg_y = by + ay
                fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y, mode='lines',
                    line=dict(color=arrow_colour, width=arrow_width),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
        if vline:
            fig.add_vline(x=np.log10(vline), line_dash='dash', line_color='grey')
    
        # Axis limits, trimmed by percentile the same way the matplotlib version did
        xaxis_kwargs = dict(title=xlabel)
        yaxis_kwargs = dict(title=ylabel)
        if log_x:
            xaxis_kwargs['type'] = 'log'
        if log_y:
            yaxis_kwargs['type'] = 'log'
        if axis_perc > 0 and y is not None:
            xaxis_kwargs['range'] = [axis_lim(x, min, perc=axis_perc),
                                    axis_lim(x, max, perc=axis_perc)]
            yaxis_kwargs['range'] = [axis_lim(y, min, perc=axis_perc),
                                    axis_lim(y, max, perc=axis_perc)]
        # Explicit fixed range (e.g. +/- search radius for astrometry offset plots) —
        # takes priority over axis_perc, and is required whenever equal_aspect is used,
        # since autorange combined with scaleanchor is unreliable in Kaleido
        if xlim is not None:
            xaxis_kwargs['range'] = list(xlim)
        if ylim is not None:
            yaxis_kwargs['range'] = list(ylim)
        if reverse_x:
            xaxis_kwargs['autorange'] = 'reversed'
            xaxis_kwargs.pop('range', None)
    
        fig.update_layout(
            title=title,
            xaxis=xaxis_kwargs,
            yaxis=yaxis_kwargs,
            template='plotly_white',
            width=800, height=800,
            showlegend=bool(leg_labels) or (extra_traces is not None
                                            and any(tr.get('name') for tr in extra_traces)),
        )
    
        if equal_aspect:
            fig.update_yaxes(scaleanchor='x', scaleratio=1)
    
        # Annotated stats text block (replaces matplotlib plt.text)
        if text is not None:
            tx, ty, xa, ya = self._loc_to_paper_xy(loc)
            fig.add_annotation(xref='paper', yref='paper', x=tx, y=ty,
                            xanchor=xa, yanchor=ya, text=text.replace('\n', '<br>'),
                            showarrow=False, align=xa,
                            bgcolor='rgba(255,255,255,0.7)')
    
        return fig
 

    def plot(self, x, y=None, c=None, yerr=None, figure=None, arrows=None, line_funcs=None,
             title='', labels=None, text=None, reverse_x=False, xlabel='', ylabel='',
             clabel='', leg_labels='', handles=[], loc='bl', ellipses=None, axis_perc=10,
             filename='screen', redo=False, vline=None, extra_traces=None, 
             log_x=False, log_y=False, equal_aspect=False, xlim=None, ylim=None):

        """Create and write a plot of the data. Plotly handles every output mode:
        'html' writes an interactive file, 'screen' opens in a browser tab, and any
        other extension ('png', 'pdf', 'eps', etc) is rendered via kaleido.
    
        Arguments and keyword arguments mirror build_plotly_figure(). See that
        docstring for details on each.
    
        Returns:
        --------
        None"""

        # Only write figure if user wants it
        if not self.write:
            return
            
        thumb = None
        
        if filename != 'screen':
            # Derive name of thumbnail file
            thumb = '{0}_thumb.png'.format(filename[:-1-len(self.plot_to)])
        
        # Don't produce plot if file exists and user didn't specify to re-do
        if filename != 'screen' and os.path.exists(filename) and not redo:
            if self.verbose:
                print('File already exists. Skipping plot.')
        else:
            fig = self.build_plotly_figure(
                x, y=y, c=c, yerr=yerr, clabel=clabel, title=title, xlabel=xlabel, ylabel=ylabel,
                labels=labels, line_funcs=line_funcs, ellipses=ellipses, arrows=arrows,
                leg_labels=leg_labels, axis_perc=axis_perc, vline=vline, extra_traces=extra_traces,
                log_x=log_x, log_y=log_y, reverse_x=reverse_x, text=text, loc=loc, 
                equal_aspect=equal_aspect, xlim=xlim, ylim=ylim
            )
    
            if self.verbose:
                print("Writing figure to '{0}'.".format(filename))
    
            if thumb is not None:
                fig.write_image(thumb, scale = 0.1)
    
            if filename == 'screen':
                fig.show()
            elif 'html' in filename:
                fig.write_html(filename, include_plotlyjs='cdn', full_html=True)
            else:
                w_in, h_in = self.fig_size['figsize']
                fig.write_image(filename, width=int(w_in * 100), height=int(h_in * 100))
            
        if filename != 'screen':
            self.html.write(self.add_html_link(filename, thumb))


    def validate(self, name1, name2, redo=False):

        """Produce a validation report between two catalogues, and optionally produce plots.

        Arguments:
        ----------
        name1 : string
            The dictionary key / name of a catalogue from the main catalogue object used to
            compare other data.
        name2 : string
            The dictionary key / name of a catalogue from the main catalogue object used as
            a comparison.

        Keyword Arguments:
        ------------------
        redo: bool
            Produce this plot and write it, even if the file already exists.

        Returns:
        --------
        ratio_med : float
            The median flux density ratio. -1 if this is not derived.
        sep_med : float
            The median sky separation between the two catalogues.
        alpha_med : float
            The median spectral index. -1 if this is not derived."""

        print('Validating {0} with {1}...'.format(name1, name2))

        filename = 'screen'

        # Write survey and number of matched to cross-matches report table
        self.html.write("""<tr>
                        <td>{0}</td>
                        <td>{1}</td>
                        <td>{2}""".format(name2, self.cat.freq[name2], self.cat.count[name2]))

        # Plot the positional offsets
        title = "{0} \u2014 {1} positional offsets".format(name1, name2)
        if self.plot_to != 'screen':
            filename = '{0}/{1}_{2}_astrometry.{3}'.format(self.figDir, name1, name2, self.plot_to)

        # Compute the S/N and its log based on main catalogue
        if name1 in list(self.cat.flux.keys()):
            self.cat.df['SNR'] = self.cat.flux[name1] / self.cat.flux_err[name1]
            self.cat.df['logSNR'] = np.log10(self.cat.df['SNR'])
            caxis = 'logSNR'
        else:
            caxis = None

        # Get non-nan data shared between each used axis as a numpy array
        x, y, c, indices = self.shared_indices(self.cat.dRA[name2],
                                               yaxis=self.cat.dDEC[name2], caxis=caxis)

        # Derive the statistics of x and y and store in string to annotate on figure
        dRAmed, dRAmean, dRAstd, dRAerr, dRAmad = get_stats(x)
        dDECmed, dDECmean, dDECstd, dDECerr, dDECmad = get_stats(y)
         # Format labels according to destination of figure
        if self.plot_to == 'html':
            txt = 'Median \u0394RA: %.2f\n' % dRAmed
            txt += 'Mean \u0394RA: %.2f\n' % dRAmean
            txt += '\u03C3(\u0394RA): %.2f\n' % dRAstd
            txt += '\u03C3(mean \u0394RA): %.2f\n' % dRAerr
            txt += 'Median \u0394DEC: %.2f\n' % dDECmed
            txt += 'Mean \u0394DEC: %.2f\n' % dDECmean
            txt += '\u03C3(\u0394DEC): %.2f\n' % dDECstd
            txt += '\u03C3(mean \u0394DEC): %.2f' % dDECerr
            
        else:
            txt = r'$\widetilde{\Delta RA}$: %.2f' % dRAmed + '\n'
            txt += r'$\overline{\Delta RA}$: %.2f' % dRAmean + '\n'
            txt += r'$\sigma_{\Delta RA}$: %.2f' % dRAstd + '\n'
            txt += r'$\sigma_{\overline{\Delta RA}}$: %.2f' % dRAerr + '\n'
            txt += r'$\widetilde{\Delta DEC}$: %.2f' % dDECmed + '\n'
            txt += r'$\overline{\Delta DEC}$: %.2f' % dDECmean + '\n'
            txt += r'$\sigma_{\Delta DEC}$: %.2f' % dDECstd + '\n'
            txt += r'$\sigma_{\overline{\Delta DEC}}$: %.2f' % dDECerr


        # Create an ellipse at the position of the median with axes of standard deviation
        e1 = {'center': (dRAmed, dDECmed), 'width': dRAstd, 'height': dDECstd,
              'color': 'black'}

        # Force axis limits of the search radius
        radius = max(self.cat.radius[name1], self.cat.radius[name2])

        # Create an ellipse at 0,0 with width 2 x search radius
        e2 = {'center': (0, 0), 'width': radius * 2, 'height': radius * 2,
              'color': 'grey', 'dash': 'dash'}

        # Format labels according to destination of figure
        if self.plot_to == 'html':
            xlabel = '\u0394RA (arcsec)'
            ylabel = '\u0394DEC (arcsec)'
            clabel = 'log\u2081\u2080 S/N'
        else:
            xlabel = r'$\Delta$RA (arcsec)'
            ylabel = r'$\Delta$DEC (arcsec)'
            clabel = r'$\log_{10}$ S/N'

        # For html plots, add S/N and separation labels for every data point
        if caxis is not None:
            labels = ['S/N = {0:.2f}, separation = {1:.2f}\"'.format(cval, totSep) for cval,
                      totSep in zip(self.cat.df.loc[indices, 'SNR'], self.cat.sep[name2][indices])]
        else:
            labels = ['Separation = {0:.2f}\"'.format(cval) for cval in
                      self.cat.sep[name2][indices]]

        # Get median separation in arcsec
        c1 = SkyCoord(ra=0, dec=0, unit='arcsec, arcsec')
        c2 = SkyCoord(ra=dRAmed, dec=dDECmed, unit='arcsec, arcsec')
        sep_med = c1.separation(c2).arcsec

        # Get mad of separation in arcsec
        c1 = SkyCoord(ra=0, dec=0, unit='arcsec, arcsec')
        c2 = SkyCoord(ra=dRAmad, dec=dDECmad, unit='arcsec, arcsec')
        sep_mad = c1.separation(c2).arcsec

        # Write the dRA and dDEC to html table
        self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f} (RA)<br>{2:.2f} &plusmn {3:.2f} (Dec)
                        <br>""".format(dRAmed, dRAmad, dDECmed, dDECmad))

        # Plot the positional offsets
        self.plot(x,
                  y=y,
                  c=c,
                  line_funcs=(self.x0, self.y0),
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  clabel=clabel,
                  text=txt,
                  ellipses=(e1, e2),
                  axis_perc=0,
                  loc='tr',
                  filename=filename,
                  labels=labels,
                  redo=redo,
                  equal_aspect=True,
                  xlim=(-radius, radius),
                  ylim=(-radius, radius))

        # Plot the positional offsets across the sky
        title += " by sky position"
        xlabel = 'RA (deg)'
        ylabel = 'DEC (deg)'
        if self.plot_to != 'screen':
            filename = '{0}/{1}_{2}_astrometry_sky.{3}'.format(self.figDir, name1, name2,
                                                               self.plot_to)

        # Get non-nan data shared between each used axis as a numpy array
        x, y, c, indices = self.shared_indices(self.cat.ra[name2], yaxis=self.cat.dec[name2],
                                               caxis=caxis)

        # For html plots, add S/N and separation labels for every data point
        if caxis is not None:
            labels = ['S/N = {0:.2f}, \u0394RA = {1:.2f}\", \u0394DEC = {2:.2f}\"'.format(cval, dra,
                                                                                          ddec)
                      for cval, dra, ddec in zip(self.cat.df.loc[indices, 'SNR'],
                                                 self.cat.dRA[name2][indices],
                                                 self.cat.dDEC[name2][indices])]
        else:
            labels = ['\u0394RA = {0:.2f}\", \u0394DEC = {1:.2f}\"'.format(dra, ddec) for dra, ddec
                      in zip(self.cat.dRA[name2][indices], self.cat.dDEC[name2][indices])]

        # Plot the positional offsets across the sky
        self.plot(x,
                  y=y,
                  c=c,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  reverse_x=True,
                  arrows=(self.cat.dRA[name2][indices], self.cat.dDEC[name2][indices]),
                  clabel=clabel,
                  axis_perc=0,
                  filename=filename,
                  labels=labels,
                  redo=redo,
                  equal_aspect=True)

        # Derive column names and check if they exist
        freq = int(round(self.cat.freq[name1]))
        fitted_flux_col = '{0}_extrapolated_{1}MHz_flux'.format(name2, freq)
        fitted_ratio_col = '{0}_extrapolated_{1}MHz_{2}_flux_ratio'.format(name2, freq, name1)
        ratio_col = '{0}_{1}_flux_ratio'.format(name2, name1)

        # Only plot flux ratio if it was derived
        if ratio_col not in self.cat.df.columns and (fitted_ratio_col not in self.cat.df.columns
                                                     or np.all(np.isnan(
                                                         self.cat.df[fitted_ratio_col]))):
            print("Can't plot flux ratio since you haven't derived the fitted flux density "
                  "at this frequency.")
            ratio_med = -111
            ratio_mad = -111
            flux_ratio_type = ''
            self.html.write('<td>')
        else:
            # Compute flux ratio based on which one exists and rename variable for figure title
            if ratio_col in self.cat.df.columns:
                ratio = self.cat.df[ratio_col]
                flux_ratio_type = name2
            elif fitted_ratio_col in self.cat.df.columns:
                ratio = self.cat.df[fitted_ratio_col]
                flux_ratio_type = '{0}-extrapolated'.format(name2)

            logRatio = np.log10(ratio)

            # Plot the flux ratio as a function of S/N

            title = "{0} / {1} flux ratio".format(name1, flux_ratio_type)
            xlabel = 'S / N'
            ylabel = 'Flux Density Ratio'
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_ratio.{3}'.format(self.figDir, name1, name2, self.plot_to)

            # Get non-nan data shared between each used axis as a numpy array
            x, y, c, indices = self.shared_indices('SNR', yaxis=ratio)

            # Derive the ratio statistics and store in string to append to plot
            ratio_med, ratio_mean, ratio_std, ratio_err, ratio_mad = get_stats(y)
             # Format labels according to destination of figure
            if self.plot_to == 'html':
                txt = 'Median Ratio: %.2f\n' % ratio_med
                txt += 'Mean Ratio: %.2f\n' % ratio_mean
                txt += '\u03C3(Ratio): %.2f\n' % ratio_std
                txt += '\u03C3(Mean Ratio): %.2f\n' % ratio_err
            else:
                txt = r'$\widetilde{Ratio}$: %.2f' % ratio_med + '\n'
                txt += r'$\overline{Ratio}$: %.2f' % ratio_mean + '\n'
                txt += r'$\sigma_{Ratio}$: %.2f' % ratio_std + '\n'
                txt += r'$\sigma_{\overline{Ratio}}$: %.2f' % ratio_err
            # For html plots, add flux labels for every data point
            if flux_ratio_type == name2:
                labels = ['{0} flux = {1:.2f} mJy, {2} flux = {3:.2f} mJy'.format(name1, flux1,
                                                                                  name2, flux2)
                          for flux1, flux2 in zip(self.cat.flux[name1][indices]*1e3,
                                                  self.cat.flux[name2][indices]*1e3)]
            else:
                labels = ['{0} flux = {1:.2f} mJy, {2} flux = {3:.2f} mJy'.format(name1, flux1,
                                                                                  flux_ratio_type,
                                                                                  flux2)
                          for flux1, flux2 in zip(self.cat.flux[name1][indices]*1e3,
                                                  self.cat.df[fitted_flux_col][indices]*1e3)]

            # Write the ratio to html report table
            if flux_ratio_type == name2:
                type = 'measured'
            else:
                type = 'extrapolated'
            self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f} ({2})<br>""".format(ratio_med, ratio_mad, type))

            # Plot the flux ratio as a function of S/N
            self.plot(x,
                      y=y,
                      c=c,
                      line_funcs=(self.y1, self.ratio_err_min, self.ratio_err_max),
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      text=txt,
                      loc='tr',
                      axis_perc=0,
                      filename=filename,
                      labels=labels,
                      redo=redo)

            # Plot the flux ratio across the sky
            title += " by sky position"
            xlabel = 'RA (deg)'
            ylabel = 'DEC (deg)'
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_ratio_sky.{3}'.format(self.figDir, name1, name2,
                                                              self.plot_to)

            # Get non-nan data shared between each used axis as a numpy array
            x, y, c, indices = self.shared_indices(self.cat.ra[name2],
                                                   yaxis=self.cat.dec[name2], caxis=logRatio)

            # Format labels according to destination of figure
            if self.plot_to == 'html':
                clabel = 'log\u2081\u2080 Flux Ratio'
            else:
                clabel = r'$\log_{10}$ Flux Ratio'

            # For html plots, add flux ratio labels for every data point
            labels = ['{0} = {1:.2f}'.format('Flux Ratio', cval) for cval in ratio[indices]]

            # Plot the flux ratio across the sky
            self.plot(x,
                      y=y,
                      c=c,
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      clabel=clabel,
                      reverse_x=True,
                      axis_perc=0,
                      filename=filename,
                      labels=labels,
                      redo=redo,
                      equal_aspect=True)

        # Derive spectral index column name and check if exists
        si_column = '{0}_{1}_alpha'.format(name1, name2)

        if si_column not in self.cat.df.columns:
            print("Can't plot spectral index between {0} and {1}, since it was not derived.".format(
                name1, name2))
            alpha_med = -111  # null flag
            self.html.write('<td>')
        else:
            # Plot the spectral index
            title = "{0}-{1} Spectral Index".format(name1, name2)
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_spectal_index.{3}'.format(self.figDir, name1,
                                                                  name2, self.plot_to)

            # Get non-nan data shared between each used axis as a numpy array
            x, y, c, indices = self.shared_indices(si_column)

            # Format labels according to destination of figure
            freq1 = int(round(min(self.cat.freq[name1], self.cat.freq[name2])))
            freq2 = int(round(max(self.cat.freq[name1], self.cat.freq[name2])))
            if self.plot_to == 'html':
                xlabel = '\u03B1 [{0}-{1} MHz]'.format(freq1, freq2)
            else:
                xlabel = r'$\alpha_{%s}^{%s}$' % (freq1, freq2)

            # Derive the statistics of x and store in string
            alpha_med, alpha_mean, alpha_std, alpha_err, alpha_mad = get_stats(x)
            if self.plot_to == 'html':
                txt = 'Median alpha: %.2f\n' % alpha_med
                txt += 'Mean alpha: %.2f\n' % alpha_mean
                txt += '\u03C3(alpha): %.2f\n' % alpha_std
                txt += '\u03C3(Mean alpha): %.2f\n' % alpha_err
            else:
                txt = r'$\widetilde{\alpha}$: %.2f' % alpha_med + '\n'
                txt += r'$\overline{\alpha}$: %.2f' % alpha_mean + '\n'
                txt += r'$\sigma_{\alpha}$: %.2f' % alpha_std + '\n'
                txt += r'$\sigma_{\overline{\alpha}}$: %.2f' % alpha_err

            # Write the ratio to html report table
            self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f}<br>""".format(alpha_med, alpha_mad))

            # Plot the spectral index
            self.plot(x,
                      title=title,
                      xlabel=xlabel,
                      ylabel='N',
                      axis_perc=0,
                      filename=filename,
                      text=txt,
                      loc='tl',
                      redo=redo)

        # Write the end of the html report table row
        self.html.write("""</td>
                    </tr>""")

        alpha_med = self.cat.med_si
        alpha_type = '{0}'.format(name1)

        # Create dictionary of validation metrics and where they come from
        metric_val = {'Flux Ratio': ratio_med,
                      'Flux Ratio Uncertainty': ratio_mad,
                      'RA Offset': dRAmed,
                      'DEC Offset': dDECmed,
                      'Positional Offset': sep_med,
                      'Positional Offset Uncertainty': sep_mad,
                      'Spectral Index': alpha_med}

        metric_source = {'Flux Ratio': flux_ratio_type,
                         'Flux Ratio Uncertainty': flux_ratio_type,
                         'RA Offset': name2,
                         'DEC Offset': name2,
                         'Positional Offset': name2,
                         'Positional Offset Uncertainty': name2,
                         'Spectral Index': alpha_type}

        count = self.cat.count[name2]

        # Overwrite values if they are valid and come from a larger catalogue
        for key in list(metric_val.keys()):
            if count > self.metric_count[key] and metric_val[key] != -111:
                self.metric_count[key] = count
                self.metric_val[key] = metric_val[key]
                self.metric_source[key] = metric_source[key]
