## -----------------------------------------------------------------------------
## @brief Analysis of date gathered in Repeated Run
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details ...
## ------------------------------------------------------------------------------
import glob
import json
import pathlib
import statistics as stat

import matplotlib.pyplot as plt
import numpy as np

from UHFQA_Run_Plural_Gathering import fn

DIRECTORY_OF_THIS_FILE = pathlib.Path(__file__).parent

def get_latencies_first_and_ninth(path_to_master_file):

    latencies_of_first = []
    latencies_of_first_mean_err = []

    latencies_of_ninth = []
    latencies_of_ninth_mean_err = []

    for filename in glob.glob(str(path_to_master_file / 'analysis_data*.json'), recursive=True):
        #print(filename)
        with open(filename) as json_file:
            files = json.load(json_file)

            latency = files['latency']
            latencies_of_first.append(latency[0])
            latencies_of_first_mean_err.append(files['latency_mean_err'][0])

            latency = files['latency']
            latencies_of_ninth.append(latency[8])
            latencies_of_ninth_mean_err.append(files['latency_mean_err'][8])

    return latencies_of_first, latencies_of_ninth

def plot_latencies(title, HDAWGS, latencies_of_first, latencies_of_ninth, latencies_of_seventeenth, power_cycle_ends, init_ends, path_to_master_file, plotfig):
    nanoscale = 1e9

    latencies_of_first = np.asarray(latencies_of_first)*1e9
    latencies_of_ninth = np.asarray(latencies_of_ninth)*1e9
    latencies_of_seventeenth = np.asarray(latencies_of_seventeenth)*1e9

    offset = np.floor(np.min(latencies_of_first))
    latencies_of_first = np.asarray(latencies_of_first) - offset
    latencies_of_ninth = np.asarray(latencies_of_ninth) - offset
    latencies_of_seventeenth = np.asarray(latencies_of_seventeenth) - offset

    fig, ax = plt.subplots()
    plt.title(f'Latencies for Reinitialisation of {title}')

    #ax.plot(np.asarray(latencies_of_first)*nanoscale, '*', label='First of 8146 (delay)')
    #ax.plot(np.asarray(latencies_of_ninth)*nanoscale, '*', label='First of 8198 (no delay)')

    #ax.plot(latencies_of_first, '*', label='First of 8146 (delay)')
    #ax.plot(latencies_of_ninth, '*', label='First of 8198 (no delay)')
    #ax.plot(latencies_of_seventeenth, '*', label='First of 8246 (no delay)')

    ax.plot(latencies_of_first, '*', label=f'First of {HDAWGS[0]}')
    ax.plot(latencies_of_ninth, '*', label=f'First of {HDAWGS[1]}')
    #ax.plot(latencies_of_seventeenth, '*', label=f'First of {HDAWGS[2]}')

    ax.set_xlabel('Number of Run')
    ax.set_ylabel(f'Latencies above {offset} ns in ns')
    ax.set_ylim(0.0, 3.25)
    for init_end in init_ends:
        if init_end == 0:
            ax.axvline(x=init_end-0.5, ymin=0, ymax=1, color='red', label='Reinitialisation')
        else:
            ax.axvline(x=init_end-0.5, ymin=0, ymax=1, color='red')

    for power_cycle_end in power_cycle_ends:
        if power_cycle_end == 0:
            ax.axvline(x=power_cycle_end-0.5, ymin=0, ymax=1, color='black', label='Power Cycle')
        else:
            ax.axvline(x=power_cycle_end-0.5, ymin=0, ymax=1, color='black')

    ax.legend()

    fn_filename = f'Latencies_{title}'
    run_options_in_title = ''
    figname = fn(str(path_to_master_file), fn_filename, run_options_in_title)
    if plotfig:
        plt.savefig(figname, format='pdf')
        plt.show()
    plt.close(fig)

def histo_latencies(title, channelname, latencies, path_to_master_file, plotfig):
    latencies_ns = np.asarray(latencies)*1e9
    offset = np.floor(np.min(latencies_ns))
    latencies_ns = np.asarray(latencies_ns) - offset
    bins = np.arange(np.min(latencies_ns), np.max(latencies_ns), 0.01)
    plt.figure()
    plt.title(f'Latency histogram of {channelname} in {title}')
    plt.hist(latencies_ns, bins=bins)
    plt.xlabel(f'Latency above {offset} ns in ns')
    plt.show()
    plt.close()

    latencies_ps = np.asarray(latencies)*1e12
    offset = np.floor(np.min(latencies_ps))
    latencies_ps = np.asarray(latencies_ps) - offset

    plt.figure()
    binwidth = 20
    n, bins, patches = plt.hist(x=latencies_ps, bins=np.arange(min(latencies_ps), max(latencies_ps) + binwidth, binwidth))
    """
    plt.title(f"Histogram of Signal Arrival for channel {i}")
    plt.xlabel("Offset-cropped latency in ps")
    run_options_in_title = f"channel_{i}"
    fig_path = path_to_file
    figname = UHF_pg.fn(str(fig_path), "Histogram_trigger_runs", run_options_in_title)
    if save:
        plt.savefig(figname, format="pdf")
    if not save:
        #plt.show()
        pass
    plt.close("all")"""
    plt.title(f'Latency histogram of {channelname} in {title}')
    plt.xlabel(f'Latency above {offset/1e3} ns in ps')
    plt.show()

    overallmean = stat.mean(latencies_ps)
    mask_min = np.where(latencies_ps < overallmean)
    mask_max = np.where(latencies_ps > overallmean)
    ts_all_small = latencies_ps[mask_min]
    ts_all_large = latencies_ps[mask_max]
    diff = stat.mean(ts_all_large)-stat.mean(ts_all_small)

    print(f"Average difference {diff:.{0}f} +/- {20:.{1}f} ps (assumed / approximated error)")

def Intra_HD_Skew(title, device, path_to_master_file, init_ends, power_cycle_ends, latencies1, latencies1_mean_err, latencies2, latencies2_mean_err, limits):
    picoscale = 1e12
    fig, ax = plt.subplots()
    plt.title(f'Diff for channel 0/1 and 1/2 for Reinitialisation of {title}')
    ax.errorbar(np.arange(len(latencies1)), (np.asarray(latencies2) - np.asarray(latencies1))*picoscale, yerr = np.sqrt((np.asarray(latencies2_mean_err)*picoscale)**2 + (np.asarray(latencies1_mean_err)*picoscale)**2), fmt='.', label=f'First two of {device}')
    ax.set_ylim(limits)
    ax.set_xlabel('Number of Run')
    ax.set_ylabel('Diff between channel 1 and 2 in ps')
    for init_end in init_ends:
        if init_end == 0:
            ax.axvline(x=init_end-0.5, ymin=0, ymax=1, color='red', label='Reinitialisation')
        else:
            ax.axvline(x=init_end-0.5, ymin=0, ymax=1, color='red')

    for power_cycle_end in power_cycle_ends:
        if power_cycle_end == 0:
            ax.axvline(x=power_cycle_end-0.5, ymin=0, ymax=1, color='black', label='Power Cycle')
        else:
            ax.axvline(x=power_cycle_end-0.5, ymin=0, ymax=1, color='black')

    ax.legend()

    fn_filename = f'Skew_diff_in_same_HD_{title}'
    run_options_in_title = ''
    figname = fn(str(path_to_master_file), fn_filename, run_options_in_title)
    #plt.savefig(figname, format='pdf')
    #plt.show()
    plt.close(fig)

def Inter_HD_Skew(title, dev1, dev2, path_to_master_file, init_ends, power_cycle_ends, latencies_of_first, latencies_of_first_mean_err, latencies_of_ninth, latencies_of_ninth_mean_err, limits):
    picoscale = 1e12
    fig, ax = plt.subplots()
    plt.title(f'Diff for channel 0/1 of first and second HD for Reinitialisation of {title}')
    ax.errorbar(np.arange(len(latencies_of_ninth)), (np.asarray(latencies_of_ninth) - np.asarray(latencies_of_first))*picoscale, yerr = np.sqrt((np.asarray(latencies_of_ninth_mean_err)*picoscale)**2 + (np.asarray(latencies_of_first_mean_err)*picoscale)**2), fmt='o')
    ax.set_ylim(limits)
    ax.set_xlabel('Number of Run')
    ax.set_ylabel(f'Diff between channel 1 of HDs {dev1} and {dev2} in ps')
    for init_end in init_ends:
        if init_end == 0:
            ax.axvline(x=init_end-0.5, ymin=0, ymax=1, color='red', label='Reinitialisation')
        else:
            ax.axvline(x=init_end-0.5, ymin=0, ymax=1, color='red')

    for power_cycle_end in power_cycle_ends:
        if power_cycle_end == 0:
            ax.axvline(x=power_cycle_end-0.5, ymin=0, ymax=1, color='black', label='Power Cycle')
        else:
            ax.axvline(x=power_cycle_end-0.5, ymin=0, ymax=1, color='black')

    ax.legend()

    fn_filename = f'Skew_diff_across_HDs_{title}'
    run_options_in_title = ''
    figname = fn(str(path_to_master_file), fn_filename, run_options_in_title)
    #plt.savefig(figname, format='pdf')
    plt.show()
    plt.close(fig)

def create_latency_evolution(master_folder):
    path_to_master_file = DIRECTORY_OF_THIS_FILE / 'Output_ZSync_Trigger_Skew' / master_folder
    #print(path_to_master_file)

    power_cycles = 1
    initialisations_per_power_cycle = 1
    executions_per_initialisation = 1
    title = ''

    for filename in glob.glob(str(path_to_master_file / 'repeated_run_settings*.json'), recursive=True):
        with open(filename) as json_file:
            files = json.load(json_file)
            power_cycles = files['power_cycles']
            initialisations_per_power_cycle = files['initialisations_per_power_cycle']
            executions_per_initialisation = files['executions_per_initialisation']
            title = files['title']
            HDAWGS = files['HDAWGS']

    latencies_of_first = []
    latencies_of_first_mean_err = []

    latencies_of_second = []
    latencies_of_second_mean_err = []

    latencies_of_ninth = []
    latencies_of_ninth_mean_err = []

    latencies_of_tenth = []
    latencies_of_tenth_mean_err = []

    latencies_of_seventeenth = []
    latencies_of_seventeenth_mean_err = []

    latencies_of_eighteenth = []
    latencies_of_eighteenth_mean_err = []

    for filename in glob.glob(str(path_to_master_file / '*/analysis_data*.json'), recursive=True):
        #print(filename)
        with open(filename) as json_file:
            files = json.load(json_file)

            latency = files['latency']
            latencies_of_first.append(latency[0])
            latencies_of_first_mean_err.append(files['latency_mean_err'][0])

            latency = files['latency']
            latencies_of_second.append(latency[1])
            latencies_of_second_mean_err.append(files['latency_mean_err'][1])

            latency = files['latency']
            #latencies_of_ninth.append(latency[8])
            #latencies_of_ninth_mean_err.append(files['latency_mean_err'][8])
            latencies_of_ninth.append(latency[4])
            latencies_of_ninth_mean_err.append(files['latency_mean_err'][4])

            latency = files['latency']
            #latencies_of_tenth.append(latency[9])
            #latencies_of_tenth_mean_err.append(files['latency_mean_err'][9])
            latencies_of_tenth.append(latency[5])
            latencies_of_tenth_mean_err.append(files['latency_mean_err'][5])

            latency = files['latency']
            #latencies_of_seventeenth.append(latency[16])
            #latencies_of_seventeenth_mean_err.append(files['latency_mean_err'][16])
            #latencies_of_seventeenth.append(latency[12])
            #latencies_of_seventeenth_mean_err.append(files['latency_mean_err'][12])

    init_ends = np.arange(0,len(latencies_of_first), executions_per_initialisation)

    power_cycle_ends = np.arange(0, len(latencies_of_first), executions_per_initialisation*(initialisations_per_power_cycle-1))

    # all the desired plots:
    #plot_latencies(title, latencies_of_first, latencies_of_ninth, power_cycle_ends, init_ends, path_to_master_file, True)
    plot_latencies(title, HDAWGS, latencies_of_first, latencies_of_ninth, latencies_of_seventeenth, power_cycle_ends, init_ends, path_to_master_file, True)
    #plot_latencies(title, HDAWGS, latencies_of_first[18:54], latencies_of_ninth[18:54], latencies_of_seventeenth, power_cycle_ends[:1], init_ends[:18], path_to_master_file, True)


    histo_latencies(title,'first channel of 8146', latencies_of_first, path_to_master_file, True)
    histo_latencies(title, 'first channel of 8198', latencies_of_ninth, path_to_master_file, True)

    limits = -15,45
    #Intra_HD_Skew(title, '8146 (delay)', path_to_master_file, init_ends, power_cycle_ends, latencies_of_first, latencies_of_first_mean_err, latencies_of_second, latencies_of_second_mean_err, limits)
    #Intra_HD_Skew(title, HDAWGS[0], path_to_master_file, init_ends, power_cycle_ends, latencies_of_first, latencies_of_first_mean_err, latencies_of_second, latencies_of_second_mean_err, limits)

    limits = 230, 290
    #Intra_HD_Skew(title, '8198 (no delay)', path_to_master_file, init_ends, power_cycle_ends, latencies_of_ninth, latencies_of_ninth_mean_err, latencies_of_tenth, latencies_of_tenth_mean_err, limits)
    #Intra_HD_Skew(title, HDAWGS[1], path_to_master_file, init_ends, power_cycle_ends, latencies_of_ninth, latencies_of_ninth_mean_err, latencies_of_tenth, latencies_of_tenth_mean_err, limits)

    limits = 0, 100
    #Inter_HD_Skew(title, '8146 (delay)', '8198 (no delay)', path_to_master_file, init_ends, power_cycle_ends, latencies_of_first, latencies_of_first_mean_err, latencies_of_ninth, latencies_of_ninth_mean_err, limits)
    Inter_HD_Skew(title, HDAWGS[0], HDAWGS[1], path_to_master_file, init_ends, power_cycle_ends, latencies_of_first, latencies_of_first_mean_err, latencies_of_ninth, latencies_of_ninth_mean_err, limits)

    limits = 500, 850
    #Inter_HD_Skew(title, '8146 (delay)', '8246 (no delay)', path_to_master_file, init_ends, power_cycle_ends, latencies_of_first, latencies_of_first_mean_err, latencies_of_seventeenth, latencies_of_seventeenth_mean_err, limits)
    Inter_HD_Skew(title, HDAWGS[0], HDAWGS[2], path_to_master_file, init_ends, power_cycle_ends, latencies_of_first, latencies_of_first_mean_err, latencies_of_seventeenth, latencies_of_seventeenth_mean_err, limits)


if __name__ == '__main__':
    master_folder = 'RepeatedRun_20200512-201819-fullroundnormal'
    ##master_folder = 'RepeatedRun_20200513-125316-onlyUHFinclclock'
    #master_folder = 'RepeatedRun_20200513-141204-onlyUHFinclclock'
    #master_folder = 'RepeatedRun_20200513-162220-onlyHDinclclock'
    #master_folder = 'RepeatedRun_20200513-181027-PQSCandHDs'
    #master_folder = 'RepeatedRun_20200513-192554-PQSCandHDs'
    #master_folder = 'RepeatedRun_20200513-213434-PQSCnoclockandHD'
    #master_folder = 'RepeatedRun_20200513-220731-PQSConlyextclockandHDs' #### in first
    #master_folder = 'RepeatedRun_20200513-224710-PQSConlyextclockandHDs'
    #master_folder = 'RepeatedRun_20200513-234858-onlyUHFonlyextclk'
    #master_folder = 'RepeatedRun_20200514-001614-onlyUHFnoclock'
    #master_folder = 'RepeatedRun_20200514-010346-onlyUHFonlyextclk'

    master_folder = 'RepeatedRun_20200520-134442-HDsAndLotOfPowerCycles'
    #master_folder = 'RepeatedRun_20200520-151542'
    #master_folder = 'RepeatedRun_20200520-160907'
    #master_folder = 'RepeatedRun_20200520-180959'
    master_folder = 'RepeatedRun_20200525-123002-HDsAndLotOfPowerCycles'
    master_folder = 'RepeatedRun_20200525-141750'
    master_folder = 'RepeatedRun_20200525-144237'
    master_folder = 'RepeatedRun_20200525-145720'
    master_folder = 'RepeatedRun_20200525-153603'
    master_folder = 'RepeatedRun_20200605-092405'
    master_folder = 'RepeatedRun_20200605-142053'
    master_folder = 'RepeatedRun_20200605-171408'






    master_folder = 'RepeatedRun_20200608-161311-newupdateonlytwo'

    master_folder = 'RepeatedRun_20200629-104615-powercycles'
    master_folder = 'RepeatedRun_20200629-122348-PQSCandHDs'

    #master_folder = 'RepeatedRun_20200629-151844-UHF'


    create_latency_evolution(master_folder)

    master_folders = ['RepeatedRun_20200512-201819-fullroundnormal', 'RepeatedRun_20200513-125316-onlyUHFinclclock', 'RepeatedRun_20200513-141204-onlyUHFinclclock', 'RepeatedRun_20200513-162220-onlyHDinclclock', 'RepeatedRun_20200513-181027-PQSCandHDs', 'RepeatedRun_20200513-192554-PQSCandHDs', 'RepeatedRun_20200513-213434-PQSCnoclockandHD', 'RepeatedRun_20200513-220731-PQSConlyextclockandHDs','RepeatedRun_20200513-224710-PQSConlyextclockandHDs', 'RepeatedRun_20200513-234858-onlyUHFonlyextclk', 'RepeatedRun_20200514-001614-onlyUHFnoclock', 'RepeatedRun_20200514-010346-onlyUHFonlyextclk']
