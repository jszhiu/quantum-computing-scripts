## -----------------------------------------------------------------------------
## @brief Getting locations of signals of the channels
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Getting zero-crossings of all drag functions by fitting onto drag
## function
## ------------------------------------------------------------------------------

import itertools

# import pdb
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import peakutils
import scipy.odr as odr
from peakutils.plot import plot as pplot
from scipy.optimize import curve_fit

# import setting_functions as init
import UHFQA_Plural_Analysis as UHF_pa

logger = logging.getLogger()
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# pylint: disable=bad-continuation


def fn(
    fig_path, filename, run_options
):  # pylint: disable=invalid-name,locally-disabled
    """ Helper function generatring plot filename with timestamp"""
    filename = (
        fig_path
        + "/"
        + filename
        + "_"
        + str(run_options)
        + "_"
        + time.strftime("%Y%m%d-%H%M%S")
    )
    return filename + ".pdf"


def multiple_waves(
    s,
    daq,
    HDAWGS,
    UHF,
    PQSC,
    scopeModule,
    channels_list,
    wave_nodepath,
    fig_path,
    number_of_average_runs,
):
    """
    Running more than two waveforms on the already initialized devices and scope module. Call UHFQA_Run_PluralGathering.py functions for that.

    Input:  init_settings   : dictionary containing the settings to all devices
            daq             : current data acquistion run
            UHF             : device name
            PQSC            : device name
            HDAWGS          : list of device names of certain order
            channels_list   : list of channels of same length as the HDAWGS list. the nth list corresponds to the channels that should output a signal of the nth HDAWG in the latter list.
            wave_nodepath   : nodepath to which we are subscribed to
            fig_path        : generated path to output directory where figures and json files genereted in the course of the experiment should be saved to
            number_of_average_runs: number of runs in order to calculate the jitter
            scopeModule     : already initialized scope module of UHF

    Output: results_success : if gahtering data worked (bool)

    The function is set up as follows:
    1) Starting PQSC trigger execution
    2) initializing singlerun_settings of a) if to save plots and b) the run
    3) Calling get_times in UHFQA_Run_PluralGathering
    4) initializing analysis settings and running the analysis of the run in UHF_Run_PluralGathering
    """

    """
    init.seti(daq, PQSC, 'execution/enable', 1)
    # Get Trigger progress
    progress = init.getd(daq, PQSC, 'execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')

    if warmup_time:
        logger.info(f'Now warming up for {warmup_time} minutes.')
        time.sleep(warmup_time*60)

    progress = init.getd(daq, PQSC, 'execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')"""

    """
    plotshow_settings = {
        "savepeakplot"          : True,
        "save_fitted_peaks"     : True,
        "save_raw_data"         : True,
        "fig_path"              : fig_path
    }

    run_settings = {
        "n_average"             : number_of_average_runs,
        "width"                 : 7, # min distance between peaks waveform of adjacent channels
        "scope_weigth"          : 1,
        "n_segments"            : 1, # the number of segments within one accumulation? --> probably put in the loop of compared channels in here
        # also known as the historylength of the scopeModule
        # Value to use for the historylength parameter, the Scope Module will only return this number of scope records.
        "channels_list"         : channels_list
    }

    # merge the HDAWG settings and the plotting settings into all settings
    singlerun_settings = {**init_settings, **plotshow_settings, **run_settings}
    s = singlerun_settings"""

    UHF_pa.save_data(s, s["fig_path"], "settings")

    # singlerun_data, results_success = UHF_rpg.singlerun(singlerun_settings, daq, UHF, scopeModule, wave_nodepath, fig_path)

    # Get peak times and their errors.
    data, t_all, results_success = get_times(s, daq, UHF, scopeModule, wave_nodepath)

    return data, t_all, results_success


# 2) get_times, which returns the times and errors of the pulses
######################################################################################


def get_times(s, daq, UHFQA, scopeModule, wave_nodepath):
    """
    Outputs an array of the peak times and the respective errors in another array.
    Does furthermore perform several plots in called functions.

    Input:
    a dictionary s containing the following (and more) fields:

    fig_path        : path to output of figures and json files
    savepeakplot    : Save the overlapped UHF scope module in the fig_path folder
    save_fitted_peak: Save the fit of each peak waveform into the fig_path folder
    save_raw_data   : ...

    n_average       : For each sequencer setting, how many scope shots acquired to obtain the jitter
    awg_waveform_length : waveform length in AWG samples (TODO: still adjust in code to the correct sampling, although works for now)

    daq             : the data acquisition instance of the run
    UHFQA           :the string containing the device name
    scopeModule     : the ZI wrapped UHF.scope module (could be improved)
    wave_nodepath   : the nodepath to which the acquisition instance is subscribed to


    Output:
            list t_all = [t, t_err]     : all the peak times in the first entry of the list and the peak time errors in the second entry of the list (both found by fitting waveform), the shape of both entries is t = (n_average_runs, number_of_total_channels), t_err = (n_average_runs, number_of_total_channels)
            results_success             : If the gathering of data was successful (bool)
    """
    awg_waveform_length = s["awg_waveform_length"]
    fig_path = s["fig_path"]

    ts_results = []
    ts_errors_results = []
    data = []

    # Initialize Scope
    scopeModule.set("scopeModule/historylength", s["n_segments"])
    for k in range(s["n_average"]):
        if k == 0:
            t_start = time.time()

        # Sample as long as it takes to extract valid T1/T2 times
        num_valid = 0

        # Acquire two shots.
        num_acquire = s["n_segments"]
        num_target = 1
        t_whilestart = time.time()
        t_max = 30  # maximal time of whileloop
        timeloop = 0

        # Update the scope
        daq.setInt("/" + str(UHFQA) + "/scopes/0/single", 1)
        #daq.sync()
        while num_valid < num_target and timeloop < t_max:

            # restart averager
            # UHF.scopeModule.set('scopeModule/averager/restart', 1)

            data, results_success = run_awg_read_scope_n_times(
                daq, UHFQA, scopeModule, wave_nodepath, num_acquire
            )

            if results_success:
                data = data[0]
                # use last shot which contains averaged information.
                x_meas, y_meas = extract_x_y(
                    data[-1]
                )  # x_meas = timetrace in sec rel. to trigger, y_meas = signal in arb. units

                x_meas = x_meas[0]  # it's twice the same anyway
                ## add the two scope outputs to have both
                y_meas = sum(y_meas)
                peak_thresh = 0.9
                peak_indices = peakutils.indexes(
                    y_meas, thres=peak_thresh, min_dist=s["width"]
                )

                if s["savepeakplot"]:
                    show_pplot(
                        daq, UHFQA, x_meas, y_meas, peak_indices, s["fig_path"], k
                    )

                a = len(peak_indices)

                channels_list = s["channels_list"]
                flattendchannelslist = list(
                    itertools.chain.from_iterable(channels_list)
                )
                b = len(flattendchannelslist)

                assert (
                    a == b
                ), "Measured amount of peaks does not coincide with assumed amount of peaks! Check that chosen HDAWG channels chosen in code coincide with the channels physically connected to one of the UHF inputs. If so, check if one of the peaks is too low to be seen by the code, and restart the HDAWG outputting that low peak."

                ydata, xdata = split_data_in_peaks(
                    peak_indices, x_meas, y_meas, awg_waveform_length
                )
                if s["awg_waveform"] == "drag":
                    ts_fit, ts_error = dragfit(
                        x_meas,
                        y_meas,
                        peak_indices,
                        s["width"],
                        awg_waveform_length,
                        s["fig_path"],
                        k,
                        save_fitted_peaks=s["save_fitted_peaks"],
                    )
                """
                if s['awg_waveform'] == 'gauss':
                    #ts_fit, ts_error = gaussfit(x_meas, y_meas, peak_indices, s["width"], awg_waveform_length, save_fitted_peaks=s['save_fitted_peaks'])
                    ts_fit, ts_error = gaussfit(x_meas, y_meas, peak_indices, s['width'], awg_waveform_length, fig_path, k, s['save_fitted_peaks'], fs_scope=1.8e9)"""

                ts_results.append(ts_fit)
                ts_errors_results.append(ts_error)

                num_valid += 1
                timeloop = time.time() - t_whilestart

            # assert or break --> is this called at any time? Isn't the one in the scope enough?
            # assert num_valid >= num_target, "After maximal time of {}, no valid data was acquired.".format(t_max)

            if k == 0:
                t_end = time.time()

    if results_success:
        t_all = [ts_results, ts_errors_results]
        return data, t_all, True
    t_all = None
    return data, t_all, False

    ##################################################
    # 3) Plotting and Saving Stuff from the run
    ##################################################


def show_pplot(daq, device, x_meas, y_meas, peak_indices, fig_path, k):
    """
    Saves peaks plot of full UHF scope run with the (multiple) input waveforms.

    Input:  daq         : current data acquisition instance
            device      : scope module device name
            x_meas      : time axis sampling points in seconds
            y_meas      : signal (voltage) axis sampling points in volts
            peak_indices: indices of peaks to insert in both x_meas and y_meas
            fig_path    : path to saving location
            k           : number of run out of the n_average runs gathered in order to obtain jitter

    """

    # clockbase = daq.getInt('/{}/clockbase'.format(device))
    # t = np.arange(-totalsamples, 0)*dt + (timestamp - triggertimestamp)/float(clockbase)
    fig = plt.figure(figsize=(10, 6))
    pplot(1e6 * x_meas, y_meas, peak_indices)
    plt.title("Identified Peaks on {}".format(device))
    plt.ylabel("Amplitude [V]")
    plt.xlabel("t [us]")
    fn_filename = "peaksplot"
    run_options_in_title = "run_{}".format(k)
    figname = fn(fig_path, fn_filename, run_options_in_title)
    plt.savefig(figname, format="pdf")
    # plt.show()
    plt.close("all")


def dragfit(
    x_meas,
    y_meas,
    peak_indices,
    g_width,
    awg_waveform_length,
    fig_path,
    k,
    save_fitted_peaks=False,
    fs_scope=1.8e9,
):
    """ Providing a Drag fit to the two peaks

    Input:  x_meas      : time axis sampling points in seconds
            y_meas      : signal (voltage) axis sampling points in volts
            peak_indices: indices of peaks to insert in both x_meas and y_meas
            g_width     : min distance between peaks waveform of adjacent channels
            awg_waveform_length :
            fig_path    :
            k           :
            save_fitted_peaks :
            fs_scope    :

    Output: ts_fit      : fitted peak positions
            ts_error    : error residuals from fit
    """
    num_peaks = len(peak_indices)
    ts_fit = np.zeros(num_peaks)  # fitted peak positions.
    ts_error = np.zeros(num_peaks)  # error residuals.

    def dragfct(x, a, x0, sig, D):  # pylint: disable=invalid-name,locally-disabled
        return a * np.exp(-((x - x0) ** 2) / (2 * sig ** 2)) * (-2 * (x - x0)) + D

    def dragfct_odr(B, x):
        [a, x0, sig, D] = B
        return a * np.exp(-((x - x0) ** 2) / (2 * sig ** 2)) * (-2 * (x - x0)) + D

    def perform_odr(x, y, xerr, yerr, a0, x0, sig0, D0):
        """Finds the ODR for data {x, y} and returns the result"""
        drag = odr.Model(dragfct_odr)
        mydata = odr.Data(x, y, wd=1.0 / xerr, we=1.0 / yerr)
        myodr = odr.ODR(mydata, drag, beta0=[a0, x0, sig0, D0])
        odr_output = myodr.run()
        return odr_output

    for i in range(num_peaks):
        index_from = peak_indices[i] - awg_waveform_length
        index_to = peak_indices[i] + awg_waveform_length
        # We fit to data in ns to avoid any problems dealing with small numbers.

        fitdata = np.array(
            [x_meas[index_from:index_to] * 1e9, y_meas[index_from:index_to]]
        )
        a0 = y_meas[peak_indices[i]]  # pylint: disable=invalid-name,locally-disabled
        x0 = (
            x_meas[peak_indices[i]] * 1e9
        )  # pylint: disable=invalid-name,locally-disabled
        sig0 = (
            g_width / fs_scope
        ) * 1e9  # pylint: disable=invalid-name,locally-disabled
        D0 = 0
        popt, pcov = curve_fit(
            dragfct,
            fitdata[0],
            fitdata[1],
            p0=[a0, x0, sig0, D0],
            bounds=((0, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)),
        )
        n_points = 1000

        #######################################
        # odr
        #######################################

        a0, x0, sig0, D0 = popt

        x_i_err = (fitdata[0][1] - fitdata[0][0]) * 0.5
        y_i_err = max(fitdata[1]) * 0.05

        regression = perform_odr(
            fitdata[0], fitdata[1], x_i_err, y_i_err, a0, x0, sig0, D0
        )

        # print (regression.beta)

        x0_odr = regression.beta
        err = regression.sd_beta

        if save_fitted_peaks:
            fig = plt.figure()
            plt.plot(fitdata[0], fitdata[1], "b+:", label="data")
            plot_x_axis = np.linspace(fitdata[0, 0], fitdata[0, -1], n_points)
            plt.axvline(x=popt[1])
            plt.text(
                fitdata[0, 0] + 0.3 * (popt[1] - fitdata[0, 0]),
                popt[0] / 2.0,
                str("{0:3f} ns".format(popt[1])),
            )
            # plt.plot(plot_x_axis, dragfct(plot_x_axis, *popt), 'r-', label='fit')
            plt.plot(plot_x_axis, dragfct(plot_x_axis, *x0_odr), "r-", label="fit")
            plt.xlabel("Time, relative to trigger (ns)")
            plt.ylabel("Signal (a.u.)")
            plt.legend()
            fn_filename = "drag_fit"
            run_options_in_title = "run_{}_channel_{}".format(k, i)
            figname = fn(fig_path, fn_filename, run_options_in_title)
            plt.savefig(figname, format="pdf")
            plt.close("all")

        # Reconvert results from ns to s in the odr case
        ts_fit[i] = x0_odr[1] * 1e-9  # Estimate for x0.
        ts_error[i] = err[1] * 1e-9  # Std for x0.

        perr = np.sqrt(np.diag(pcov))
        # Reconvert results from ns to s.
        # ts_fit[i] = popt[1]*1e-9 # Estimate for x0.
        # ts_error[i] = perr[1]*1e-9 # Std for x0.

    return ts_fit, ts_error


"""
def gaussfit(x_meas, y_meas, peak_indices, g_width, awg_waveform_length, fig_path, k, save_fitted_peaks=False, fs_scope=1.8e9):
    ''' Providing a Gaussian fit to the peaks in the data.

    Output: ts_fit: fitted peak positions
            ts_error: error residuals
    '''
    num_peaks = len(peak_indices)
    #label = ["Left Peak", "Right Peak"]
    ts_fit = np.zeros(num_peaks) # fitted peak positions.
    ts_error = np.zeros(num_peaks) # error residuals.

    def gaussian(x, a, x0, sig, D):
        return a*np.exp(-((x-x0)**2)/(2*sig**2))+ D
    # diff_index = peak_indices[1] - peak_indices[0]

    def gaussian_odr(B, x):
        [a, x0, sig, D] = B
        return a*np.exp(-((x-x0)**2)/(2*sig**2)) + D

    def perform_odr(x, y, xerr, yerr, a0, x0, sig0, D0):
        '''Finds the ODR for data {x, y} and returns the result'''
        gauss = odr.Model(gaussian_odr)
        mydata = odr.Data(x, y, wd=1./xerr, we=1./yerr)
        myodr = odr.ODR(mydata, gauss, beta0=[a0, x0, sig0, D0])
        odr_output = myodr.run()
        return odr_output

    for i in range(num_peaks):

        index_from = peak_indices[i] - awg_waveform_length
        index_to = peak_indices[i] + awg_waveform_length
        # We fit to data in ns to avoid any problems dealing with small numbers.
        fitdata = np.array([x_meas[index_from:index_to]*1e9, y_meas[index_from:index_to]])

        a0 = y_meas[peak_indices[i]]
        x0 = x_meas[peak_indices[i]]*1e9
        sig0 = (g_width/fs_scope)*1e9
        D0 = 0


        ###############################################################
        # curvefit
        ###############################################################
        popt, pcov = curve_fit(gaussian, fitdata[0], fitdata[1], p0=[a0, x0, sig0, D0], bounds=((0, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)))
        n_points = 1000
        ##############################################################
        # ODR
        ##############################################################
        a0, x0, sig0, D0 = popt

        x_i_err = (fitdata[0][1]-fitdata[0][0])*0.25
        y_i_err = max(fitdata[1])*0.05

        regression = perform_odr(fitdata[0], fitdata[1], x_i_err, y_i_err, a0, x0, sig0, D0)

        x0_odr = regression.beta
        err = regression.sd_beta

        if save_fitted_peaks:
            #save_gaussian_plot(i, k, fitdata, popt, gaussian, n_points, fig_path)
            save_gaussian_plot(i, k, fitdata, x0_odr, gaussian, n_points, fig_path)
         # from //https://scipy.github.io/devdocs/generated/scipy.optimize.curve_fit.html

        perr = np.sqrt(np.diag(pcov))

        # Reconvert results from ns to s in the odr case
        ts_fit[i] = x0_odr[1]*1e-9 # Estimate for x0.
        ts_error[i] = err[1]*1e-9 # Std for x0.

        # for the curve_fit case
        # ts_fit[i] = popt[1]*1e-9 # Estimate for x0.
        # ts_error[i] = perr[1]*1e-9 # Std for x0.

    return ts_fit, ts_error


def save_gaussian_plot(i, k, fitdata, popt, gaussian, n_points, fig_path):
    ''' Save data and gaussian fit for channel i in run k '''
    fig = plt.figure()
    plt.plot(fitdata[0], fitdata[1], 'b+:', label='data')
    plot_x_axis = np.linspace(fitdata[0, 0], fitdata[0, -1], n_points)
    plt.axvline(x=popt[1])
    plt.text(fitdata[0, 0]+0.3*(popt[1]-fitdata[0, 0]), popt[0]/2., str("{0:3f} ns".format(popt[1])))
    plt.plot(plot_x_axis, gaussian(plot_x_axis, *popt), 'r-', label='fit')
    plt.xlabel('Time, relative to trigger (ns)')
    plt.ylabel('Signal (a.u.)')
    #plt.show()
    fn_filename = 'gauss_fit'
    run_options_in_title = 'run_{}_channel_{}'.format(k, i)
    figname = fn(fig_path,  fn_filename, run_options_in_title)
    plt.savefig(figname, format='pdf')
    plt.close('all')
"""


def run_awg_read_scope_n_times(daq, UHF, scopeModule, wave_nodepath, n_segments):
    """
    Reads scope and returns data dictionary.
    Checks flags and cleans data in case invalid data has been returned
    """
    scopeModule.set("scopeModule/clearhistory", 1)
    daq.setInt("/%s/scopes/0/enable" % UHF, 1)
    scopeModule.execute()

    start = time.time()
    records = 0

    # This will be measured in first run
    t_per_record = 0.05
    time_estimate = t_per_record * n_segments
    timeout = max(time_estimate * 10, 30)
    time.sleep(t_per_record * n_segments + 0.01)

    # Wait until the Scope Module has received and processed the desired number of records.
    while records < n_segments or (time.time() - start) > timeout:
        # records = zi_shell_device_uhf.scopeModule.getInt("scopeModule/records")
        records = scopeModule.getInt("scopeModule/records")
        time.sleep(t_per_record)
        # print("Scope module has acquired {} records (requested {})".format(records, n_segments), end='\r')
        if (time.time() - start) > timeout:
            # Break out of the loop if for some reason we're no longer receiving scope data from the device.
            print(
                "\nScope Module did not return {} records after {} s - forcing stop.".format(
                    n_segments, timeout
                )
            )
            break

    # Read out the scope data from the module.
    data = scopeModule.read(True)
    # disable scope
    daq.setInt("/%s/scopes/0/enable" % UHF, 0)
    # Stop the module.
    scopeModule.finish()

    if wave_nodepath not in data:
        print(
            "The Scope Module did not return data for {}. Check the UHF Trigger.".format(
                wave_nodepath
            )
        )
        return None, False

    data = data[wave_nodepath]
    # Clean data and output the flags
    # data = check_and_clean_data_flags(data)
    return data, True


def extract_x_y(
    data, f_s_scope=1.8e9
):  # x_meas = timetrace in sec rel. to trigger, y_meas = signal in arb. units
    """Converst scopedata to plottable x-y traces
    # In: f_s_scope: sampling rate of Scope
    # Out: x_measured: Timetrace in s
    #      y_measured: Volatege in (V) TODO: Output values seem unreasonable
    """
    x_measured = []
    y_measured = []
    for l in range(
        0, len(data["channelenable"])
    ):  # pylint: disable=invalid-name,locally-disabled
        p = data["channelenable"][l]  # pylint: disable=invalid-name,locally-disabled
        if p:
            y_measured.append(data["wave"][l])
            x_measured.append(
                np.arange(-data["totalsamples"], 0) * data["dt"]
                + (data["timestamp"] - data["triggertimestamp"]) / f_s_scope
            )
    return x_measured, y_measured

def split_data_in_peaks(peak_indices, x_meas, y_meas, awg_waveform_length):
    """ given the peak indices of a peak pair, split the data into two portions each containing one peak """
    ydata = []
    xdata = []
    for peak_index in peak_indices:
        y = y_meas[
            peak_index - awg_waveform_length : peak_index + awg_waveform_length
        ]  # still have to change by linking the length of the wave in the HDAWG with the length of the wave in the UHF scope (dependent on the chosen sampling frequencies)
        x = x_meas[peak_index - awg_waveform_length : peak_index + awg_waveform_length]
        ydata.append(y)
        xdata.append(x)
    return ydata, xdata
