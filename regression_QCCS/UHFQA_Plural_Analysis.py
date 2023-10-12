## -----------------------------------------------------------------------------
## @brief Calculation of latency, skew and jitter for the trigger and feedback tests
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details data_analysis function calling the other functions to output required
## into json file
## ------------------------------------------------------------------------------

import itertools
import json
import logging

# import pdb
# import os
import pathlib
import statistics as stat
import time

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

import UHFQA_Run_Plural_Gathering as UHF_pg
import PeaksDiffsHistogram

logger = logging.getLogger()
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# pylint: disable=no-member
# pylint: disable=bad-continuation


def ndarray_to_list(obj):
    """ Convert ndarray to list """
    return obj.tolist()


def save_data(results_data, fig_path, filename):
    """ Save results array as json file """
    new_filename = filename + time.strftime("%Y%m%d-%H%M%S") + ".json"
    filename = fig_path + "/" + new_filename
    with open(filename, "w") as outfile:
        json.dump(
            results_data, outfile, default=ndarray_to_list
        )  # convert ndarrays to lists


"""
def maxDiff(a, a_err):
    '''
    Calculates the maximum difference between any two entries of a vector, and assigns the error, which is the error of the two
    entries added.
    '''
    vmin = a[0]
    vmin_err = a_err[0]
    dmax = 0
    for i, a_i in enumerate(a):
        if a[i] < vmin:
            vmin = a[i]
            vmin_err = a_err[i]
        elif a[i] - vmin > dmax:
            dmax = a[i] - vmin
            dmax_err = a_err[i] + vmin_err
    return dmax, dmax_err"""


def maxDiff(a, a_err):
    """
    Calculates the maximum difference between any two entries of a vector, and assigns the error, which is the error of the two
    entries added.
    """
    arg_max = np.argmax(a)
    arg_min = np.argmin(a)

    dmax = a[arg_max] - a[arg_min]
    dmax_err = a_err[arg_max] + a_err[arg_min]

    return dmax, dmax_err


def maxDiff_averror(a, a_err):
    """
    Calculates the maximum difference between any two entries of a vector, and assigns the avg error.
    """
    arg_max = np.argmax(a)
    arg_min = np.argmin(a)

    dmax = a[arg_max] - a[arg_min]
    dmax_err = (a_err[arg_max] + a_err[arg_min]) / 2

    return dmax, dmax_err


def calculate_and_plot_jitter_each_channel(
    ts, ts_err, channels_list, HDAWGS, gap_wait, fig_path, nanoscale
):
    """
    For each channel, a plot showing the ts with the error ts_err is done. During this loop over all channels, the channelnumber
    calculated.
    Then, the jitter for each channel is calculated, in two ways: the std dev of the same channel and the maximal difference between two entries of the same channel. There is again a plot, and all the data is also returned. For both ways of jitter calculation, there is a potential, not seen error that can come from the measuring device, notably the UHF.
    """
    # define std-deviation, max difference in two points and mean
    std = []
    std_err = []

    maxdiff = []
    maxdiff_err = []

    mean = []
    # latency = []
    # unnormalized_latency_8146_channel_0 = []
    mean_err = []

    channelnumber = 0
    for channels_list_i, HDAWG in zip(channels_list, HDAWGS):
        for i, channel in enumerate(channels_list_i):
            ts_i = ts[
                :, i
            ]  # extracting all peaks of channel 'channel' which is number i in list
            ts_err_i = ts_err[:, i]

            std.append(stat.stdev(ts_i))
            ksum = np.sqrt(sum((ts_i - stat.mean(ts_i)) ** 2))
            ss = np.sqrt(1 / (len(ts_i) - 1)) * ksum
            dell_x = (
                np.sqrt(1 / (len(ts_i) - 1))
                * 1.0
                / 2
                * 1
                / ksum
                * 2
                * (ts_i - stat.mean(ts_i))
                * ts_err_i
            )
            s_err = np.sqrt(sum(dell_x ** 2))
            std_err.append(s_err)

            mean.append(stat.mean(ts_i))
            # mean takes the std.dev error of all the initial errors, and this is therefore smaller than the initial errors
            mean_err.append(1.0 / len(ts_i) * np.sqrt(sum(np.asarray(ts_err_i) ** 2)))

            maxd, maxd_err = maxDiff(ts_i, ts_err_i)
            maxdiff.append(maxd)
            maxdiff_err.append(maxd_err)

            channelnumber += 1

            fig = plt.figure()
            plt.title("Zero Crossing channel {} in {} ".format(channel, HDAWG))
            plt.errorbar(
                np.arange(len(ts_i)), ts_i * nanoscale, yerr=ts_err_i * nanoscale
            )
            plt.ticklabel_format(useOffset=False, style="plain")
            plt.xlabel("number of run")
            plt.ylabel("position of zero crossing in time [ns]")
            fn_filename = "channel_msrmts"
            run_options_in_title = "HDAWG_{}_channel_{}".format(HDAWG, channel)
            figname = UHF_pg.fn(fig_path, fn_filename, run_options_in_title)
            plt.savefig(figname, format="pdf")
            plt.close("all")

            # latency

        if HDAWG != HDAWGS[-1]:
            ts = ts[:, len(channels_list_i) :]

    # Jitter for each channel
    picoscale = 1e12

    std = np.asarray(std)
    std_err = np.asarray(std_err)

    maxdiff = np.asarray(maxdiff)
    maxdiff_err = np.asarray(maxdiff_err)

    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.errorbar(
        np.arange(len(std)), std * picoscale, yerr=std_err * picoscale, fmt="o"
    )
    plt.xlabel("Channel number")
    plt.ylabel("Jitter in std deviation [ps]")

    plt.subplot(2, 1, 2)
    plt.errorbar(
        np.arange(len(maxdiff)),
        maxdiff * picoscale,
        yerr=maxdiff_err * picoscale,
        fmt="o",
    )
    plt.xlabel("Channel number")
    plt.ylabel("Jitter in max Diff [ps]")
    fn_filename = "Jitter"
    run_options_in_title = ""
    figname = UHF_pg.fn(fig_path, fn_filename, run_options_in_title)
    fig.suptitle("Jitter for each channel \n " + run_options_in_title)
    plt.savefig(figname, format="pdf")
    plt.close("all")

    # latency = [x - (1.5*n+1.5)*gap_wait*0.555555555*1e-9 for n, x in enumerate(mean)] # wrong!!!
    # latency = [x - (2*n+1.5)*gap_wait/(2.4e9) for n, x in enumerate(mean)] # with the awg_sampling_rate 2.4e9
    # unnormalized_latency_of_first_channel_for_all_runs = ts_i

    return std, std_err, maxdiff, maxdiff_err, mean, mean_err, channelnumber


"""
def calculate_latencies():

    # latency = [x - (1.5*n+1.5)*gap_wait*0.555555555*1e-9 for n, x in enumerate(mean)] # wrong!!!
    latency = [
        x - (2 * n + 1.5) * gap_wait / (2.4e9) for n, x in enumerate(mean)
    ]  # with the awg_sampling_rate 2.4e9
    unnormalized_latency_of_first_channel_for_all_runs = ts_i
    return latency, unnormalized_latency_of_first_channel_for_all_runs
"""


def plot_latency_of_each_channel(latency, mean_err, fig_path, nanoscale):
    mean = np.asarray(latency)
    mean_err = np.asarray(mean_err)

    plt.figure()
    plt.errorbar(
        np.arange(len(mean)), mean * nanoscale, yerr=mean_err * nanoscale, fmt="o"
    )
    plt.title("Waveform position latencies compared to scope trigger")
    plt.xlabel("Channel number")
    plt.ylabel("Mean Latency Time of channel zero crossing [ns]")
    fn_filename = "Latency"
    run_options_in_title = ""
    figname = UHF_pg.fn(fig_path, fn_filename, run_options_in_title)
    plt.savefig(figname, format="pdf")
    plt.close("all")


def set_matrix_plot(
    mat1,
    label,
    fn_filename,
    fig_path,
    HDAWGS,
    channels_list,
    channels_ends,
    scale="large",
    save=True,
):
    """
    Configuring the matrix plot for skew or jitter matrix and saving it into fig_path. Should not be used outside the data_analysis function because it uses parameters defined only in data_analysis and not input into the function (sorry).

    Input:  mat1            : to-be-plotted matrix
            label           : labels used in the plot
            fn_filename     : filename to which should be saved to

    Output: None
    """
    picoscale = 1e12
    mat = mat1 * picoscale
    vmin = np.min(mat)
    vmax = np.max(mat)
    mask = np.where(mat == 0)
    mat[mask] = None
    fig, ax = plt.subplots(figsize=(8, 9))

    """
    colorm = cmap=plt.get_cmap('RdYlGn_r')
    if scale == 'large':
        divnorm = colors.DivergingNorm(vmin=0, vcenter=200, vmax=1000)
    elif scale == 'small':
        divnorm = colors.DivergingNorm(vmin=0, vcenter=80, vmax=200)
    else:
        print('no possible scale...')
    """

    if scale == "very_large":
        divnorm = colors.DivergingNorm(vmin=0, vcenter=420, vmax=2100)
        colorm1 = cmap1 = plt.cm.Reds(np.linspace(0.2, 1, 128))
        colorm2 = cmap2 = plt.cm.summer(np.linspace(0.0, 1, 128))

        colorm = np.vstack((colorm2, colorm1))
        colorm = colors.LinearSegmentedColormap.from_list("my_colormap", colorm)

    elif scale == "large":
        divnorm = colors.DivergingNorm(vmin=0, vcenter=200, vmax=1000)
        colorm1 = cmap1 = plt.cm.Reds(np.linspace(0.2, 1, 128))
        colorm2 = cmap2 = plt.cm.summer(np.linspace(0.0, 1, 128))

        colorm = np.vstack((colorm2, colorm1))
        colorm = colors.LinearSegmentedColormap.from_list("my_colormap", colorm)

    elif scale == "small":
        divnorm = colors.DivergingNorm(vmin=0, vcenter=80, vmax=200)
        colorm1 = cmap1 = plt.cm.Reds(np.linspace(0.3, 1, 128))
        colorm2 = cmap2 = plt.cm.summer(np.linspace(0.0, 1, 128))
        colorm = np.vstack((colorm2, colorm1))
        colorm = colors.LinearSegmentedColormap.from_list("my_colormap", colorm)

    elif scale == "tiny":
        divnorm = colors.DivergingNorm(vmin=0, vcenter=20, vmax=60)
        colorm1 = cmap1 = plt.cm.Reds(np.linspace(0.3, 1, 128))
        colorm2 = cmap2 = plt.cm.summer(np.linspace(0.0, 1, 128))
        colorm = np.vstack((colorm2, colorm1))
        colorm = colors.LinearSegmentedColormap.from_list("my_colormap", colorm)
    else:
        print("no possible scale...")

    ax.matshow(mat, cmap=colorm, norm=divnorm)
    ax.xaxis.tick_top()
    ax.set_xlabel("Channel i within HD Device")
    ax.set_ylabel("Channel j within HD Device")

    channels_list = np.asarray(channels_list).flatten()
    plt.xticks(np.arange(len(mat[0])), channels_list)
    plt.yticks(np.arange(len(mat[0])), channels_list)

    devstr = ""
    for HDAWG in HDAWGS:
        devstr += HDAWG
        devstr += ", "
    plt.title("{} Matrix for {}".format(label, devstr))
    for channel_end in channels_ends:
        ax.axhline(y=0.5 + channel_end, xmin=0, xmax=1, color="black")
        ax.axvline(x=0.5 + channel_end, ymin=0, ymax=1, color="black")
    cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap=colorm), ax=ax)
    cbar.set_label("{} [ps]".format(label), rotation=270)
    if scale == "large":
        cbar.ax.plot(np.linspace(0, 1, 100), [200 / 1000] * 100, color="black")
    elif scale == "small":
        cbar.ax.plot(np.linspace(0, 1, 100), [80 / 200] * 100, color="black")
    ##############################
    # fn_filename = 'Skew_Matrix'
    run_options_in_title = ""
    """
    for HDAWG, channels_list_i in zip(HDAWGS, channels_list):
        channelstring = ''.join(str(e) for e in channels_list_i)
        run_options_in_title += '{}_channels_{}___'.format(HDAWG, channelstring)
    """
    figname = UHF_pg.fn(fig_path, fn_filename, run_options_in_title)
    if save:
        plt.savefig(figname, format="pdf")
    else:
        plt.show()
    plt.close("all")


def plot_latency(
    latency,
    mean_err,
    fig_path,
    HDAWGS,
    channels_list,
    channels_ends,
    nanoscale,
    save=True,
):
    # channels_list = np.asarray(np.hstack(channels_list))
    channels_list = list(itertools.chain.from_iterable(channels_list))
    fig, ax = plt.subplots()
    ax.errorbar(
        np.arange(len(latency)), np.asarray(latency) * nanoscale, yerr=mean_err, fmt="o"
    )
    plt.xticks(np.arange(len(latency)), channels_list)
    for channel_end in channels_ends:
        ax.axvline(x=0.5 + channel_end, ymin=0, ymax=1, color="black")
    ax.set_xlabel("Channelnumber within HD Device")
    ax.set_ylabel("Latency of Channel in ns")

    devstr = ""
    for HDAWG in HDAWGS:
        devstr += HDAWG
        devstr += ", "
    plt.title("Latencies for {}".format(devstr))
    run_options_in_title = ""
    figname = UHF_pg.fn(fig_path, "Latencies", run_options_in_title)
    if save:
        plt.savefig(figname, format="pdf")
    else:
        plt.show()
    plt.close("all")


def calculate_skew_and_jitter_matrices(
    ts, ts_err, channels_list, HDAWGS, gap_wait, fig_path, nanoscale, channelnumber, awg_sampling_rate
):
    """
    In order to not account for any error introduced by the measuring device (UHFQA), the jitter and skew is not calculated for each channel over all runs, but for each run, and the reference is therefore not dependent on the UHFQA.
    Therefore, in a loop, the difference for each two channel zero crossings ts_i, ts_j, is calculated at first. The respective error is the added error.
    diff_expected is the variable of the expected time difference between ts_i and ts_j. This takes into account that for channels being n channels apart, the distance should be 1.5*gap_wait*0.5555 ns, since this is the HD's AWG rate. The skew is then the difference between expected and measured value, and this skew number is given for each run.

    The mean of each of these differences for every ts_i and ts_j is the skew, and is placed into mat_jitter_mean. The error is the std deviation in them, and saved into mat_jitter_std. This is a estimation of the error being low.

    The jitter is calculated as the maximal difference between two runs showing the time difference of ts_j, ts_i. So whichever run has the largest difference in the two channels. The error is calculated to be the error with the same corresponding index, and comes from the addition of the errors that were coming from the fit on the points ts,  namely ts_err.
    """
    picoscale = 1e12

    skew = []
    skew_mean = []
    skew_var = []

    latency = []
    latency_mean_error = []
    unnormalized_latency_8146_channel_0 = []

    counter = 0

    channels = []
    for l, (channels_list_i, HDAWG) in enumerate(zip(channels_list, HDAWGS)):
        for i, channel in enumerate(channels_list_i):

            # for l, _, HDAWG in enumerate(channels_list, HDAWGS):
            # for k, _ in enumerate(channels_list[l]):
            # channels.append(channels_list[l][k]+8*l) #TODO: this 8 could be a problem... for the time being: only one with 4 at the end
            channels.append(channels_list_i[i] + 8 * l)

    channels_ends = []
    previous = 0
    for l, _ in enumerate(channels_list):
        channels_ends.append(len(channels_list[l]) - 1 + previous)
        previous += len(channels_list[l])

    mat_jitter_mean = np.zeros((channelnumber, channelnumber))
    mat_jitter_std = np.zeros((channelnumber, channelnumber))

    mat_jitter_maxdiff = np.zeros((channelnumber, channelnumber))
    mat_jitter_maxdiff_err = np.zeros((channelnumber, channelnumber))
    for i in range(0, channelnumber):
        ts_i = np.mean(ts[:, i])
        lat_i_err = stat.stdev(ts[:, i])
        lat_i = ts_i - channels[i] * 2 * gap_wait / (awg_sampling_rate) - 1.5 * gap_wait / (awg_sampling_rate)
        latency.append(lat_i)
        latency_mean_error.append(lat_i_err)
        for j in range(i + 1, channelnumber):
            """
            vars()['diff_meas_{}_{}'.format(i,j)] = ts[:, j]-ts[:, i]
            vars()['diff_meas_{}_{}_err'.format(i,j)] = ts_err[:, j]+ts_err[:, i]
            multi = (channels[j]-channels[i])*2*gap_wait
            diff_expected = multi/(2.4e9)
            skew_l = diff_expected - vars()['diff_meas_{}_{}'.format(i,j)]
            skew.append(skew_l)

            mat_jitter_mean[i,j] = np.abs(np.mean(skew_l))
            mat_jitter_std[i,j] = stat.stdev(skew_l)

            maxd, maxd_err = maxDiff(skew_l, vars()['diff_meas_{}_{}_err'.format(i,j)])
            mat_jitter_maxdiff[i,j] = maxd
            mat_jitter_maxdiff_err[i, j] = maxd_err
            """
            ts_ij_diff = ts[:, j] - ts[:, i]
            ts_ij_diff_err = ts_err[:, j] + ts_err[:, i]
            # pdb.set_trace()
            diff_expected = (channels[j] - channels[i]) * 2 * gap_wait / (awg_sampling_rate)
            skew_l = diff_expected - ts_ij_diff
            skew.append(skew_l)

            mat_jitter_mean[i, j] = np.abs(np.mean(skew_l))
            mat_jitter_std[i, j] = stat.stdev(skew_l)

            # arg_max = np.argmax(skew_l)
            # maxd = skew_l[arg_max]
            # maxd_err = ts_ij_diff_err[arg_max]

            maxd, maxd_err = maxDiff_averror(skew_l, ts_ij_diff_err)
            mat_jitter_maxdiff[i, j] = maxd
            mat_jitter_maxdiff_err[i, j] = maxd_err

    skew = np.asarray(skew)
    skew_mean = np.mean(skew, axis=1)
    skew_var = np.var(skew, axis=1)

    """
    for supdir, dirs, files in os.walk(fig_path):
        for file in files:
            if file.startswith('settings'):
                settings_file = file

    print (settings_file)
    with open(fig_path+'/'+settings_file, 'r') as json_file:
        data = json.load(json_file)
        HDAWGS = data['HDAWGS']
    """
    """
    plot_latency(latency, channels_ends)
    fig,ax = plt.subplots(figsize=(8,9))
    ax.plot(latency, '*')
    for channel_end in channels_ends:
        ax.axvline(x=0.5+channel_end, ymin=0, ymax=1, color='black')
    plt.show()"""

    set_matrix_plot(
        mat_jitter_mean,
        "NoDelay Skew",
        "Skew_Matrix_NoDelay",
        fig_path,
        HDAWGS,
        channels_list,
        channels_ends,
        scale="very_large",
        save=True,
    )
    set_matrix_plot(
        mat_jitter_mean,
        "Skew",
        "Skew_Matrix",
        fig_path,
        HDAWGS,
        channels_list,
        channels_ends,
        scale="large",
        save=True,
    )
    set_matrix_plot(
        mat_jitter_std,
        "Skew Fluctuations in Std",
        "SkewFluct_Std_Matrix",
        fig_path,
        HDAWGS,
        channels_list,
        channels_ends,
        scale="small",
    )
    set_matrix_plot(
        mat_jitter_maxdiff,
        "Skew Fluctations in MaxDiff",
        "SkewFluct_Max_Diff_Matrix",
        fig_path,
        HDAWGS,
        channels_list,
        channels_ends,
        scale="small",
    )
    set_matrix_plot(
        mat_jitter_maxdiff_err,
        "Skew Fluctations in MaxDiff Error",
        "SkewFluct_Max_Diff_Matrix_Err",
        fig_path,
        HDAWGS,
        channels_list,
        channels_ends,
        scale="small",
    )

    plot_latency(
        latency,
        latency_mean_error,
        fig_path,
        HDAWGS,
        channels_list,
        channels_ends,
        nanoscale,
    )

    return (
        skew_mean,
        mat_jitter_mean,
        mat_jitter_std,
        mat_jitter_maxdiff,
        mat_jitter_maxdiff_err,
        latency,
        latency_mean_error,
        channels_ends,
    )


def data_analysis(singlerun_data, a_s, analysis_fig_path=None):
    """
    Configures data analysis run in a multiple_waves run and saves analysis output.

    Input:  singlerun_data  : to-be-analysed data in a dictionary containing ts_all being the times of a peak appearance and ts_all_error the error in these peaks. The ts_all is a list of shape (n_avg, n_ch) with n_avg runs that have been performed to deduce the jitter and n_ch number of total channels which are examined.
            a_s             : dictionary containing all the info accumulated during experiment initialization and run, some of which are needed in the course of this function

    Output: None

    The function extracts the peak times, and performs the following analysis:
    1) Plotting the measured peak times for each channel for the number_of_average_runs = n_avg → out of which jitter will be deduced
    2) One plot lining up the averaged peak times (Locations of peaks in time).
    3) A jitter plot showing the jitter as the std deviation from 1) as well as the maximum time distance measured in 1). The second is probably the better metric for jitter. This jitter includes the jitter of the UHF.
    4) The skew matrix showing the skew between any combination of chosen HDAWG channels (Skew_Matrix)
    5) The two skew fluctuation matrices. Once the matrix showing the std deviation of the skew and once the matrix showing the maximal difference in peaks. This can be interpreted as a form of jitter. Because the UHF's jitter is not included in this comparison, it serves as a good estimation of inter-channel jitter of the HDAWG only.
    6) The found analysis data is saved into a json file.

    """
    nanoscale = 1e9

    # Output of channels
    multiplegaussrun_data = singlerun_data
    ts = multiplegaussrun_data["ts_all"]
    ts_err = multiplegaussrun_data["ts_all_error"]

    ts = np.asarray(ts)
    ts_err = np.asarray(ts_err)

    channels_list = multiplegaussrun_data["channels_list"]
    HDAWGS = multiplegaussrun_data["HDAWGS"]
    gap_wait = multiplegaussrun_data["gap_wait"]
    awg_sampling_rate_value = multiplegaussrun_data["awg_sampling_rate_value"]
    fig_path = multiplegaussrun_data["fig_path"]
    if analysis_fig_path is not None:
        fig_path = analysis_fig_path

    (
        std,
        std_err,
        maxdiff,
        maxdiff_err,
        mean,
        mean_err,
        channelnumber,
    ) = calculate_and_plot_jitter_each_channel(
        ts, ts_err, channels_list, HDAWGS, gap_wait, fig_path, nanoscale
    )

    (
        skew_mean,
        mat_jitter_mean,
        mat_jitter_std,
        mat_jitter_maxdiff,
        mat_jitter_maxdiff_err,
        latency,
        latency_mean_error,
        channels_ends,
    ) = calculate_skew_and_jitter_matrices(
        ts, ts_err, channels_list, HDAWGS, gap_wait, fig_path, nanoscale, channelnumber, awg_sampling_rate_value
    )

    # plot_latency_of_each_channel(latency, mean_err, fig_path, nanoscale)

    analysis_data_results = {
        "maxdiff": maxdiff,
        "maxdiff_err": maxdiff_err,
        "peak_locations": skew_mean,
        "mean_err": mean_err,
        "latency": latency,
        "latency_mean_err": latency_mean_error,
        #'unnormalized_latency_of_first_channel_for_all_runs': unnormalized_latency_of_first_channel_for_all_runs,
        "channels_ends": channels_ends,
        "jitter_max_diff": maxdiff,
        "jitter_max_diff_err": maxdiff_err,
        "jitter_std": std,
        "jitter_std_err": std_err,
        "skew_matrix": mat_jitter_mean,
        "skew_fluct_std_matrix": mat_jitter_std,
        "skew_fluct_maxdiff_matrix": mat_jitter_maxdiff,
        "skew_fluct_maxdiff_matrix_err": mat_jitter_maxdiff_err,
    }

    # analysis_data_results = {**analysis_data_results}

    if a_s["save_analysis_data"]:
        save_data(analysis_data_results, fig_path, "analysis_data")

    arg = np.argmax(maxdiff)
    J = maxdiff[arg]
    J_err = maxdiff_err[arg]
    # J =  np.max(mat_jitter_maxdiff.flatten())
    # assert J < 50e-9, "Huge jitter above 50 ns, probably inconsistency within run."
    # assert J < 2e-9, "Huge jitter above 2 ns, probably inconsistency within run."
    # assert J < 400e-12, "Huge jitter above 400 ps, probably inconsistency within run."
    if J > 50 - 9:
        logger.warning(
            f"Standalone jitter is above 50 ns, with max {J*1e12:.{3}} +/- {J_err*1e12:.{3}} ps. Average can not be taken as reference."
        )
    if J < 80e-12:
        logger.info(
            f"Standalone jitter is below 80 ps, with max {J*1e12:.{3}} +/- {J_err*1e12:.{3}} ps."
        )
    else:
        logger.info(
            f"Standalone jitter is above 80 ps, with max {J*1e12:.{3}} +/- {J_err*1e12:.{3}} ps."
        )

    latency = np.asarray(latency)

    arg = np.argmax(latency)
    lat_max = latency[arg]
    lat_max_err = mean_err[arg]

    arg = np.argmin(latency)
    lat_min = latency[arg]
    lat_min_err = mean_err[arg]

    logger.info(
        f"Latency is between {lat_min*1e9:.{5}} +/- {lat_min_err*1e9:.{3}} ns and {lat_max*1e9:.{5}} +/- {lat_max_err*1e9:.{3}} ns (std dev error)."
    )

    arg = np.argmax(mat_jitter_maxdiff.flatten())
    J = mat_jitter_maxdiff.flatten()[arg]
    J_err = mat_jitter_maxdiff_err.flatten()[arg]
    # J =  np.max(mat_jitter_maxdiff.flatten())
    if J < 80e-12:
        logger.info(
            f"Matrix jitter is below 80 ps, with max {J*1e12:.{3}} +/- {J_err*1e12:.{3}} ps (avg error of compared data points)."
        )
    if J > 150e-12:
        logger.info(
            f"Matrix jitter is above 150 ps, with max {J*1e12:.{3}} +/- {J_err*1e12:.{3}} ps (avg error of compared data points). A histogram will be plotted."
        )
        PeaksDiffsHistogram.plotdiff(fig_path, save=True)
    else:
        logger.info(
            f"Matrix jitter is above 80 ps, with max {J*1e12:.{3}} +/- {J_err*1e12:.{3}} ps (avg error of compared data points)."
        )

    arg = np.argmax(mat_jitter_mean.flatten())
    S = mat_jitter_mean.flatten()[arg]
    S_err = mat_jitter_std.flatten()[arg]
    # S = np.max(mat1.flatten())
    if S < 420e-12:
        logger.info(
            f"Matrix skew is below 420 ps, with max {S*1e12:.{5}} +/- {S_err*1e12:.{3}} ps (std dev error)."
        )
    else:
        logger.info(
            f"Matrix skew is above 420 ps, with max {S*1e12:.{5}} +/- {S_err*1e12:.{3}} ps (std dev error)."
        )


DIRECTORY_OF_THIS_FILE = pathlib.Path(__file__).parent


def data_anlysis(multiplerun_data, s, analysis_fig_path=None):
    multiplerun_data = {**multiplerun_data, **s}

    # TODO(Elisa): Move somewhere else
    # if s['save_raw_data']:
    #     UHF_pa.save_data(multiplerun_data, s['fig_path'], "raw_data")

    analysis_settings = {
        "save_jitter_plot": True,
        "save_peak_center_plot": True,
        "save_skew_matrix": True,
        "save_analysis_data": True,
    }

    data_analysis(multiplerun_data, analysis_settings, analysis_fig_path)


def load_settings(filename):
    with open(filename, "r", encoding="utf-8") as infile:
        settings = json.load(infile)
    return settings


if __name__ == "__main__":

    # FIRST ONE
    """
    #data, ts_all, ts_all_error = load_data(filename=DIRECTORY_OF_THIS_FILE / 'raw_data20200424-160133.json')
    foldersto = 'Output_ZSync_Trigger_Skew/20200424-155733multiple----------------------------------'
    fileee = 'raw_data20200424-160133.json'
    fileee2 = 'settings20200424-160007.json'
    multiplerun_data = load_settings(filename=DIRECTORY_OF_THIS_FILE / foldersto / fileee)
    s = load_settings(filename=DIRECTORY_OF_THIS_FILE / foldersto / fileee2)
    analysis_fig_path = str(DIRECTORY_OF_THIS_FILE / foldersto)

    data_anlysis(multiplerun_data, s, analysis_fig_path) """

    # Second One
    """
    foldersto = 'Output_ZSync_Trigger_Skew/20200424-155151multiple'
    fileee = 'raw_data20200424-155542.json'
    fileee2 = 'settings20200424-155426.json'
    multiplerun_data = load_settings(filename=DIRECTORY_OF_THIS_FILE / foldersto / fileee)
    s = load_settings(filename=DIRECTORY_OF_THIS_FILE / foldersto / fileee2)
    analysis_fig_path = str(DIRECTORY_OF_THIS_FILE / foldersto)

    data_anlysis(multiplerun_data, s, analysis_fig_path) """
