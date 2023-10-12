## -----------------------------------------------------------------------------
## @brief Continuous run of triggering and feedback test
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Wrapping Plural_Run.py in a countinuous run, with the possibility of
## deciding which parts of the setup should be reinitialized vefore redoing the
## experiment.
## ------------------------------------------------------------------------------
import json
import logging
import os
import os.path
import pathlib
import statistics as stat
import time

import matplotlib.pyplot as plt
import numpy as np

import analysis_helpers
import init_normal_sync
import init_partial_initialisations
import powerSwitch as powerswitch
import RepeatedRunLatencies
#import zhinst.ziPython as zi
#import zhinst.utils as utils
#import UHFQA_Plural_Analysis as UHF_pa
import UHFQA_Run_Plural_Gathering as UHF_rpg
from setting_HDAWGs import set_HDAWG_channels
from UHFQA_Plural_Analysis import data_anlysis, maxDiff, save_data
from UHFQA_Run_Plural_Gathering import fn
import Run_Plural

#import temp_data_anlysis as da

# pylint: disable=redefined-builtin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def generate_repeated_run_directory(output_root):
    #''' Generates Output Directory'''
    folder_id = 'RepeatedRun_'+time.strftime("%Y%m%d-%H%M%S")
    folder_path = output_root / folder_id
    # Create new directory.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path, folder_id

def do_power_cycle(power_cycle, power_cycles):
    logger.info(f'Starting power cycle {power_cycle+1} out of {power_cycles}.')
    # TODO: take this out of the function and more abstract
    #devices = ['dev10004', 'dev2107', 'dev2333', 'dev8198', 'dev8146']
    devices = ['dev10004', 'dev2107','dev2333', 'dev8146', 'dev8186', 'dev8246', 'dev8218']
    hosts = ['powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com']
    ports = [0,1,2,3,4,5,6]

    #devices = ['dev10004', 'dev2107','dev2333', 'dev8146', 'dev8246']
    #hosts = ['powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com','powerswitch26.zhinst.com']
    #ports = [0,1,2,3,5]
    for device, host, port in zip(devices, hosts, ports):
        psw = powerswitch.powerSwitch(host)
        psw.off(port) # turn off switch 'port' (numbered from 0)
        time.sleep(0.2)
    logger.info('Switched off all devices. Waiting for them to properly shut down...')
    time.sleep(40)
    for device, host, port in zip(devices, hosts, ports):
        psw = powerswitch.powerSwitch(host)
        psw.on(port) #turn on switch 'port'
        time.sleep(0.2)
    logger.info('Switched on all devices. Waiting for them to properly start up...')
    time.sleep(200)
    """
    for device in devices:
        success = False
        if success == True:
            continue
        success = ziDeviceUtils.is_discoverable(device, timeout=90)
    """
    #time.sleep(3*60)

def run_Time_Evolution(reinit_type, timegap, repetitions, initialisations, power_cycles, PQSC, UHFQA, UHF_ext_clock, HDAWGS, channels_list_temp, number_of_average_runs, warmup_time, SERVER):
    '''
    Runs the TestRun.py for _repetitions_ amount with a certain _timegap_ inbetween each run. Calling multiple_waves function in TestRun.py.

    Outputs gathered info for further use in the main function.

    Inputs: timegap         : timegap between each RepeatedRun
            repetitions     : number of measurement repetitions
            PQSC            : device name
            UHFQA           : device name
            HDAWGS          : list of device names of certain order
            channels_list_temp   : list of channels of same length as the HDAWGS list. the nth list corresponds to the channels that should output a signal of the nth HDAWG in the latter list.
            waveform        : to-be-output signal waveform. Either 'gauss' or 'drag'
            number_of_average_runs: number of runs in order to calculate the jitter

    Outputs: folder_path    : path to the RepeatedRun folder
            daq             : initialized data acquisition run
    '''
    channels_list = []
    for channels_list_i in channels_list_temp:
        channels_list_i.sort()
        channels_list.append(channels_list_i)

    # Generate output directory.
    filehead, tail = os.path.split(__file__) # path to current folder
    DIRECTORY_OF_THIS_FILE = pathlib.Path(__file__).parent
    OUTPUT_ROOT = DIRECTORY_OF_THIS_FILE / 'Output_ZSync_Trigger_Skew' # setting the output folder
    folder_path, folder_id = generate_repeated_run_directory(OUTPUT_ROOT)

    reinit_device, reinit_settings = reinit_type

    repeated_run_settings = {
        "power_cycles"                      : power_cycles,
        "initialisations_per_power_cycle"   : initialisations,
        "executions_per_initialisation"     : repetitions,
        "execution_timegap"                 : timegap,
        "warmup_time"                       : warmup_time,
        "title"                             : reinit_device+'_'+reinit_settings,
        "folder_path"                       : str(folder_path),
        "HDAWGS"                            : HDAWGS
    }

    save_data(repeated_run_settings, str(folder_path), "repeated_run_settings")

    regression = 2

    plotshow_settings = {
        "savepeakplot"          : True,
        "save_fitted_peaks"     : True,
        "save_raw_data"         : True,
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

    latencies1 = []
    latencies9 = []
    print = logger.debug

    for power_cycle in range(power_cycles):
        #do_power_cycle(power_cycle, power_cycles)
        ###################################
        # initialise
        #####################################
        logger.info(f'Starting initialisation {1} out of {initialisations} (in power cycle {power_cycle+1}/{power_cycles}).')
        daq, scopeModule, wave_nodepath, init_settings = init_normal_sync.init_zsync_trigger_experiment(PQSC, UHFQA, UHF_ext_clock, HDAWGS, regression, SERVER)

        #######################################################
        for i, HDAWG in enumerate(HDAWGS):
            set_HDAWG_channels(daq, HDAWG, channels_list[i], regression)

        # Update the scope
        daq.setInt(f'{UHFQA}/scopes/0/single', 1)
        #daq.sync()

        for initialisation in range(initialisations-1):

            daq.setInt(f'{PQSC}/execution/enable', 1)
            # Get Trigger progress
            progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
            logger.debug(f'{progress:.{2}f} % Trigger Progress')

            daq.setInt(f'{UHFQA}/scopes/0/single', 1)
            #daq.sync()

            scopeModule.subscribe(wave_nodepath)
            scopeModule.execute()
            daq.setInt('/%s/scopes/0/enable' % UHFQA, 1)

            if warmup_time:
                logger.info(f'Now warming up for {warmup_time} minutes.')
                time.sleep(warmup_time*60)

            progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
            logger.debug(f'{progress:.{2}f} % Trigger Progress')
            #daq.sync()

            ########################################################
            for i in range(repetitions):
                logger.info(f'Starting execution {i+1} out of {repetitions} (in power cycle {power_cycle+1}/{power_cycles}, initialisation {initialisation+1}/{initialisations-1}).')
                start = time.time()

                to_figpath = str(time.strftime("%Y%m%d-%H%M%S"))+'multiple'
                fig_path = folder_path / to_figpath
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)

                logger.addHandler(logging.FileHandler(str(fig_path)+'/test.log', 'a'))
                fig_path_json = {"fig_path": str(fig_path)}

                init_settings = {**init_settings, **plotshow_settings, **run_settings, **fig_path_json}
                s = init_settings

                data, t_all, results_success = UHF_rpg.multiple_waves(init_settings, daq, HDAWGS, UHFQA, PQSC, scopeModule, channels_list, wave_nodepath, str(fig_path), number_of_average_runs)

                ZSyncInfo = analysis_helpers.helper_hdawg_get_zsync_dlycal(daq, HDAWGS)

                multiplerun_data = {
                    'data'                  : data,
                    'ts_all'                : t_all[0],
                    'ts_all_error'          : t_all[1],
                    'ZSyncInfo'             : ZSyncInfo
                    }

                if s['save_raw_data']:
                    save_data(multiplerun_data, s['fig_path'], "raw_data")

                if not results_success:
                    logger.info('No results could be printed.')
                    return

                # run data analysis
                data_anlysis(multiplerun_data, s)
                logger.info('Results have been printed.')

                latency1, latency9 = RepeatedRunLatencies.get_latencies_first_and_ninth(fig_path)

                assert 240e-9 < latency1[0] , f"Latency in first channel of first HD not in wanted range 240 ns <  latency < 270 ns: {latency1[0]*1e9:.{2}f} ns < 240 ns"
                assert latency1[0] < 270e-9, f"Latency in first channel of first HD not in wanted range 240 ns <  latency < 270 ns: {latency1[0]*1e9:.{2}f} ns > 270 ns"

                assert 240e-9 < latency9[0] , f"Latency in first channel of second HD not in wanted range 240 ns <  latency < 270 ns: {latency9[0]*1e9:.{2}f} ns < 240 ns"
                assert latency1[0] < 270e-9, f"Latency in first channel of second HD not in wanted range 240 ns <  latency < 270 ns: {latency9[0]*1e9:.{2}f} ns > 270 ns"

                latencies1.append(latency1)
                latencies9.append(latency9)

                maxdiff1, _ = maxDiff(np.asarray(latencies1), np.zeros(len(latencies1)))
                maxdiff9, _ = maxDiff(np.asarray(latencies9), np.zeros(len(latencies9)))

                maxdiff1 = maxdiff1[0]
                maxdiff9 = maxdiff9[0]

                #assert maxdiff1 < 200e-12, f"Difference in measured latencies between executions in first channel of first HD is too large: {maxdiff1*1e9:.{2}f} ns > 0.2 ns"
                #assert maxdiff9 < 200e-12, f"Difference in measured latencies between executions in first channel of second HD is too large: {maxdiff9*1e9:.{2}f} ns > 0.2 ns"

                if timegap:
                    end = time.time()
                    diff = end - start
                    still_to_timegap = timegap - diff
                    time.sleep(still_to_timegap)

            # If running: disable the PQSC trigger execution
            daq.setInt(f'/{PQSC}/execution/enable', 0)
            #daq.setInt('/%s/scopes/0/enable' % UHFQA, 0)
            #daq.sync()

            logger.info(f'Starting initialisation {reinit_settings} on {reinit_device} number {initialisation+2} out of {initialisations} (in power cycle {power_cycle+1}/{power_cycles}).')

            # first variant: reinitialising the whole setup
            if reinit_device == 'Whole_Setup':
                Run_Plural.disconnect(daq, PQSC, HDAWGS, UHFQA)
                daq, scopeModule, wave_nodepath, init_settings = init_normal_sync.init_zsync_trigger_experiment(PQSC, UHFQA, UHF_ext_clock, HDAWGS, regression, SERVER)

            # second variant: changing the UHFQA
            elif reinit_device == 'UHFQA':
                daq.setInt('/%s/scopes/0/enable' % UHFQA, 0)
                #daq.sync()
                wave_nodepath = init_partial_initialisations.init_zsync_trigger_experiment_UHFQA(daq, UHFQA, scopeModule, init_settings, reinit_settings)

            # third variant: changing the HDs
            elif reinit_device == 'HDs':
                init_partial_initialisations.init_zsync_trigger_experiment_HDAWG(daq, HDAWGS, init_settings, reinit_settings)

                for i, HDAWG in enumerate(HDAWGS):
                    set_HDAWG_channels(daq, HDAWG, channels_list[i], regression)

            # fourth variant: changing the PQSC and the HDs
            elif reinit_device == 'PQSC_and_HDs':
                init_partial_initialisations.init_zsync_trigger_experiment_PQSC(daq, PQSC, init_settings, reinit_settings)
                init_partial_initialisations.init_zsync_trigger_experiment_HDAWG(daq, HDAWGS, init_settings, reinit_settings='full')
                for i, HDAWG in enumerate(HDAWGS):
                    set_HDAWG_channels(daq, HDAWG, channels_list[i], regression)

            else:
                assert True, (f'{reinit_device} is no valid reinitialisation.')

        Run_Plural.disconnect(daq, PQSC, HDAWGS, UHFQA)

    return folder_path, daq

def RepeatedRun_analysis(folder_path, timegap):
    '''
    Analyses the results of folders in a RepeatedRun measurement, loops through all folders, and calculates and outputs the Jitter and Skew over time, ouputting it into the RepeatedRun folder.

    Inputs:     folder_path     : path to the RepeatedRun folder containing all the repeated run measurement json files (and figures)
                timegap         : timegap between the repeated run measurements

    Outputs:    None
    '''

    JitterMaxAv = []
    JitterStdAv = []
    SkewMax = []
    SkewAv = []

    for supdir, dirs, files in os.walk(folder_path):
        #pdb.set_trace()
        for dir_i in dirs:
            dir_i =  str(folder_path) + '/'+ dir_i
            for supdir, dirs, files in os.walk(dir_i):
                os.chdir(dir_i)
                try:
                    for file in files:
                        if file.startswith('analysis'):
                            filename = file
                    # TODO: Abort if filename doesnt exist
                    file_name = os.path.join(os.path.dirname(__file__), filename)
                    with open(filename, 'r') as json_file:
                        data = json.load(json_file)

                    jitterMax = data["jitter_max_diff"]
                    JitterMaxAv.append(stat.mean(jitterMax))

                    jitterStd = data["jitter_std"]
                    JitterStdAv.append(stat.mean(jitterStd))

                    SkewMat = data["skew_matrix"]
                    SkewMax.append(np.max(SkewMat))

                    SkewAv.append(np.nanmean(np.where(SkewMat!=0,SkewMat,np.nan)))
                except KeyError:
                    JitterMaxAv.append(None)
                    JitterStdAv.append(None)
                    SkewMax.append(None)
                    SkewAv.append(None)
                    pass

    nanoscale = 1e9
    picoscale = 1e12
    if timegap is not None:
        minutestimegapscale = 1./60*timegap
    else:
        minutestimegapscale = 1

    JitterMaxAv = np.asarray(JitterMaxAv)
    JitterStdAv = np.asarray(JitterStdAv)
    SkewMax = np.asarray(SkewMax)
    SkewAv = np.asarray(SkewAv)

    ################## Jitter #########################
    fig = plt.figure(figsize=(8,8))
    fig.suptitle('Jitter per channel (in time order) evolving over time')
    plt.subplot(2,1,1)
    plt.plot(np.arange(len(JitterStdAv))*minutestimegapscale, JitterStdAv*picoscale, '+')
    plt.xlabel('Time Evolution [min]')
    plt.ylabel('Jitter in std deviation [ps]')

    plt.subplot(2,1,2)
    plt.plot(np.arange(len(JitterMaxAv))*minutestimegapscale, JitterMaxAv*picoscale, '+')
    plt.xlabel('Time Evolution [min]')
    plt.ylabel('Jitter in max Diff [ps]')

    fn_filename = 'Jitter_Time_Evolution'
    run_options_in_title = ''
    figname = fn(folder_path, fn_filename, run_options_in_title)
    plt.savefig(figname, format='pdf')
    plt.close(fig)
    ####################################################

    ################# Skew ###########################
    fig = plt.figure(figsize=(8,8))
    fig.suptitle('Skew evolving over time')
    plt.subplot(2,1,1)
    plt.plot(np.arange(len(SkewAv))*minutestimegapscale, SkewAv*picoscale, '+')
    plt.xlabel('Time Evolution [min]')
    plt.ylabel('Average Skew between channels [ps]')

    plt.subplot(2,1,2)
    plt.plot(np.arange(len(SkewMax))*minutestimegapscale, SkewMax*picoscale, '+')
    plt.xlabel('Time Evolution [min]')
    plt.ylabel('Maximal Skew between channels [ps]')

    fn_filename = 'Skew_Time_Evolution'
    run_options_in_title = ''
    figname = fn(folder_path, fn_filename, run_options_in_title)
    plt.savefig(figname, format='pdf')
    plt.close(fig)
    ####################################################

    RepeatedRun_data_results = {
        'JitterStdAv'       : JitterStdAv,
        'JitterMaxAv'       : JitterMaxAv,
        'SkewAv'            : SkewAv,
        'SkewMax'           : SkewMax,
        'Timegap'           : timegap
    }

    save_data(RepeatedRun_data_results, folder_path, "RepeatedRun_data_results")

if __name__ == '__main__':
    '''
    In tis main function, the timegap and repetitions of the repeated run are defined. Then, a list of the used HDAWGs as well as the channels list containing a list of channels for each HDAWG must be set. The order of the list is thereby the order of the HDAWGS in that list.
    Furthermore, Z-Sync triggering PQSC and Scope-providing UHFQA are input, and the waveform = 'drag' or waveform = 'gauss' (but don't do the gauss for now).
    Lastly, the number of runs per acquisition can be set, over which per time shot the average is taken.
    '''
    ######################### INPUT Repeated Run ######################################
    timegap = 2*60 # every 5 minutes
    executions = 165

    timegap = None # every 3 minutes
    executions = 1000 #5, 3

    initialisations = 3 #10 # has to be larger than 1
    # Have the device first, then a comma and then the settings
    reinit_device = 'Whole_Setup'
    #reinit_device = 'HDs'
    reinit_device = 'PQSC_and_HDs'
    #reinit_device = 'UHFQA'

    reinit_settings = 'full'
    #reinit_settings = 'only_ext_clock_change'
    #reinit_settings = 'no_clock_change'

    reinit_type = [reinit_device, reinit_settings]

    power_cycles = 20

    ######################### INPUT Embedded Run #############################
    # The names of all devices in the setup go here:

    HDAWGS = ['dev8146', 'dev8246', 'dev8186']
    channels_list_temp = [[0,1,2,3,4,5,6,7], [0,1,2,3], [0,1,2,3]] # if changing here, also need to change in RepeatedRun.py

    HDAWGS = ['dev8146', 'dev8246']
    channels_list_temp = [[0,1,2,3,4,5,6,7], [0,1,2,3]]

    PQSC = 'dev10021'
    UHFQA = 'dev2333'
    UHF_ext_clock = 'dev2109'
    number_of_average_runs = 10 #watch out: if increasing number of average_runs, you also need to increase the timegap of calling! -> no....?
    warmup_time = 20 # minutes
    warmup_time = 2
    warmup_time = 0.5
    SERVER = "10.42.0.229"
    SERVER = '10.42.0.228'
    SERVER = 'localhost' #127.0.0.1
    ################################################################

    #print = logger.debug
    folder_path, daq = run_Time_Evolution(reinit_type, timegap, executions, initialisations, power_cycles, PQSC, UHFQA, UHF_ext_clock, HDAWGS, channels_list_temp,  number_of_average_runs, warmup_time, SERVER)

    #folder_path = 'C://Users//elisaw//Desktop//Git//products-labone//soft//testing//regression_QCCS//Output_ZSync_Trigger_Skew//RepeatedRun_20200512-142500'
    #RepeatedRun_analysis(str(folder_path), timegap)


    # Disable everything that is connected at the time being
    """
    x = daq.getByte('/zi/devices/connected')
    ConnectedDevices = x.split(",")
    for device in ConnectedDevices:
        utils.disable_everything(daq, device)

    seti(daq, PQSC, 'execution/enable', 0)"""

    """
    #snipped from sanda:
    import utils.powerSwitch as powerswitch
    psw = powerswitch.powerSwitch(r.host, a_logger)
    psw.on(5) # turn on switch 5 (numbered from 0)
    psw.off(5) #turn off switch 5
    """
