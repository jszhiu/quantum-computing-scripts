## -----------------------------------------------------------------------------
## @brief Main run for trigger and feedback tests
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Full run, calling the initialisation script and the analysis scripts
## ------------------------------------------------------------------------------

import logging
import os
import time

import analysis_helpers
import init_feedback
#import Run_getting_and_preplotting_raw_data as preplot
import init_normal_sync
import init_ZSync_and_DIO_trigger
import setting_HDAWGs
#import zhinst.ziPython as zi
#import zhinst.utils as utils
import UHFQA_Plural_Analysis as UHF_pa
import UHFQA_Run_Plural_Gathering as UHF_rpg

# pylint: disable=redefined-builtin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# 127.0.0.1
###############################################################
def gen_output_directory(output_root, additional_to_name='', additional_folder=''):
    ''' Generates Output Directory'''
    fig_id = time.strftime("%Y%m%d-%H%M%S")+additional_to_name
    #pdb.set_trace()
    fig_path = output_root +"/" + fig_id
    if additional_folder != '':
        fig_path = output_root +"/"+ additional_folder + "/"+ fig_id
    # Create new directory.
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    return fig_path

def init_directory(filehead, additional_to_name='', additional_folder=''):
    output_root = filehead # setting the output folder
    fig_path = gen_output_directory(output_root, additional_to_name, additional_folder)
    logger.addHandler(logging.FileHandler(fig_path+'/test.log', 'a'))
    #print = logger.debug
    return fig_path

def disconnect(daq, PQSC, HDAWGS, UHFQA):
    """
    Disconnecting the connected devices and disabling the PQSC's trigger output.
    """
    # Get Trigger progress
    progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
    logger.info(f'{progress:.{2}f} % Trigger Progress')

    # Disable everything.
    for HDAWG in HDAWGS:
        daq.setInt(f'{HDAWG}/system/clocks/referenceclock/source', 0)
        logger.info(f'{HDAWG} clock set to internal')

    daq.setInt(f'{PQSC}/execution/enable', 0)
    daq.setInt(f'{PQSC}/system/clocks/referenceclock/in/source', 0)

    daq.sync()
    logger.info(f'{PQSC} execution stopped and clock set to internal')

    daq.setInt(f'{UHFQA}/system/extclk', 0)
    daq.sync()
    daq.disconnect()
    logger.info('#########################################################')

def run_main_reaction(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, channels_list_temp, number_of_average_runs, regression, warmup_time, init_experiment, SERVER):
    '''
    Do a single run of the multiple_waves function. First checking the length of HDAWGs and channels_list_temp, initializing the experiment, setting the channels and running the multiple_waves function.

    Input:  UHFQA             : device name
            PQSC              : device name
            HDAWGS            : list of device names of certain order
            channels_list_temp   : list of channels of same length as the HDAWGS list. the nth list corresponds to the channels that should output a signal of the nth HDAWG in the latter list.
            number_of_average_runs: number of runs in order to calculate the jitter
            regression        : bool showing if in regression setup or not
            warmup_time       : in case of regression, the warmup_time of running the devices before starting to measure in seconds

    Output: None

    '''

    assert len(HDAWGS) == len(channels_list_temp), "Number of HDAWGS must coincide with the input number of channel_list. If no channel of some HDAWG should be addressed, delete the HDAWG from the list."

    UHFQA = UHFQA_meas

    filehead, _ = os.path.split(__file__) # path to current folder
    filehead = os.path.dirname(os.path.abspath(__file__))
    print(f'filehead={filehead:s}')

    if init_experiment == 'feedback_DIO_trig':
        fig_path = init_directory(filehead, additional_to_name='multiple', additional_folder='/Output_ZSync_FeedbackDIO_Skew')
        daq, scopeModule, wave_nodepath, init_settings = init_feedback.init_zsync_feedbackDIO_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER)

    if init_experiment == 'simple_trigger':
        fig_path = init_directory(filehead, additional_to_name='multiple', additional_folder='/Output_ZSync_Trigger_Skew')
        daq, scopeModule, wave_nodepath, init_settings = init_normal_sync.init_zsync_trigger_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER)

    # sort channels_list s.t. correct in calling
    channels_list = []
    for channels_list_i in channels_list_temp:
        channels_list_i.sort()
        channels_list.append(channels_list_i)

    #######################################################
    for i, HDAWG in enumerate(HDAWGS):
        setting_HDAWGs.set_HDAWG_channels(daq, HDAWG, channels_list[i], regression)

    # Update the scope
    daq.setInt('/' + str(UHFQA) + '/scopes/0/single', 1)
    #time.sleep(10)
    #s = daq.get(f'/{HDAWGS[0]}/raw/stats/busy')
    #s = wait_busy(daq, f'/{HDAWGS[0]}/raw/stats/busy', perform_sync=False)
    #print (s)
    #daq.sync()
    ########################################################

    # can comment in or out if you want a preplotting seeing the difference in the UHF scopes
    # preplotting(init_settings, daq, HDAWGS, UHFQA, PQSC, scopeModule, channels_list, wave_nodepath)

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
    s = singlerun_settings

    daq.setInt(f'{PQSC}/execution/enable', 1)
    # Get Trigger progress
    progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')
    scopeModule.execute()
    daq.setInt('/%s/scopes/0/enable' % UHFQA_meas, 1)

    if warmup_time:
        logger.info(f'Now warming up for {warmup_time} minutes.')
        time.sleep(warmup_time*60)

    progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')

    data, t_all, results_success = UHF_rpg.multiple_waves(s, daq, HDAWGS, UHFQA, PQSC, scopeModule, channels_list, wave_nodepath, fig_path, number_of_average_runs)

    ZSyncInfo = analysis_helpers.helper_hdawg_get_zsync_dlycal(daq, HDAWGS)

    multiplerun_data = {
        'data'                  : data,
        'ts_all'                : t_all[0],
        'ts_all_error'          : t_all[1],
        'ZSyncInfo'             : ZSyncInfo
        }

    if s['save_raw_data']:
        UHF_pa.save_data(multiplerun_data, s['fig_path'], "raw_data")

    disconnect(daq, PQSC, HDAWGS, UHFQA)

    if not results_success:
        logger.info('No results could be printed.')
        return

    UHF_pa.data_anlysis(multiplerun_data, s)
    logger.info('Results have been printed.')

def run_qubit_reset(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, channels_list_temp, number_of_average_runs, regression, warmup_time, init_experiment, SERVER):
    '''
    Do a single run of the multiple_waves function. First checking the length of HDAWGs and channels_list_temp, initializing the experiment, setting the channels and running the multiple_waves function.

    Input:  UHFQA             : device name
            PQSC              : device name
            HDAWGS            : list of device names of certain order
            channels_list_temp   : list of channels of same length as the HDAWGS list. the nth list corresponds to the channels that should output a signal of the nth HDAWG in the latter list.
            number_of_average_runs: number of runs in order to calculate the jitter
            regression        : bool showing if in regression setup or not
            warmup_time       : in case of regression, the warmup_time of running the devices before starting to measure in seconds

    Output: None

    '''

    assert len(HDAWGS) == len(channels_list_temp), "Number of HDAWGS must coincide with the input number of channel_list. If no channel of some HDAWG should be addressed, delete the HDAWG from the list."

    UHFQA = UHFQA_meas

    filehead, _ = os.path.split(__file__) # path to current folder
    filehead = os.path.dirname(os.path.abspath(__file__))
    print(f'filehead={filehead:s}')

    fig_path = init_directory(filehead, additional_to_name='multiple', additional_folder='/Output_ZSync_Qubit_Reset')
    daq, scopeModule, wave_nodepath, init_settings = init_feedback.init_zsync_feedbackDIO_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER) # TODO: Insert here the qubit reset calling

    # sort channels_list s.t. correct in calling
    channels_list = []
    for channels_list_i in channels_list_temp:
        channels_list_i.sort()
        channels_list.append(channels_list_i)

    #######################################################
    for i, HDAWG in enumerate(HDAWGS):
        setting_HDAWGs.set_HDAWG_channels(daq, HDAWG, channels_list[i], regression)

    # Update the scope
    daq.setInt('/' + str(UHFQA) + '/scopes/0/single', 1)
    ########################################################

    # can comment in or out if you want a preplotting seeing the difference in the UHF scopes
    # preplotting(init_settings, daq, HDAWGS, UHFQA, PQSC, scopeModule, channels_list, wave_nodepath)

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
    s = singlerun_settings

    daq.setInt(f'{PQSC}/execution/enable', 1)
    # Get Trigger progress
    progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')
    scopeModule.execute()
    daq.setInt('/%s/scopes/0/enable' % UHFQA_meas, 1)

    if warmup_time:
        logger.info(f'Now warming up for {warmup_time} minutes.')
        time.sleep(warmup_time*60)

    progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')

    data, t_all, results_success = UHF_rpg.multiple_waves(s, daq, HDAWGS, UHFQA, PQSC, scopeModule, channels_list, wave_nodepath, fig_path, number_of_average_runs)
    #### TODO: Add sth to find out if the drag is inverted or not

    ZSyncInfo = analysis_helpers.helper_hdawg_get_zsync_dlycal(daq, HDAWGS)

    multiplerun_data = {
        'data'                  : data,
        'ts_all'                : t_all[0],
        'ts_all_error'          : t_all[1],
        'ZSyncInfo'             : ZSyncInfo
        }

    if s['save_raw_data']:
        UHF_pa.save_data(multiplerun_data, s['fig_path'], "raw_data")

    disconnect(daq, PQSC, HDAWGS, UHFQA)

    if not results_success:
        logger.info('No results could be printed.')
        return

    #### UHF_pa.data_anlysis(multiplerun_data, s) #TODO: Add own new analysis of found data
    logger.info('Results have been printed.')

def run_partial_reaction(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, channels_list_temp, number_of_average_runs, regression, warmup_time, init_experiment, SERVER):
    '''
    Do a single run of the multiple_waves function. First checking the length of HDAWGs and channels_list_temp, initializing the experiment, setting the channels and running the multiple_waves function.

    Input:  UHFQA             : device name
            PQSC              : device name
            HDAWGS            : list of device names of certain order
            channels_list_temp   : list of channels of same length as the HDAWGS list. the nth list corresponds to the channels that should output a signal of the nth HDAWG in the latter list.
            number_of_average_runs: number of runs in order to calculate the jitter
            regression        : bool showing if in regression setup or not
            warmup_time       : in case of regression, the warmup_time of running the devices before starting to measure in seconds

    Output: None

    '''
    assert [[0,1]] == channels_list_temp, "In UHFQA triggering test should be with channels list of [[0,1]]."

    UHFQA = UHFQA_meas

    filehead, _ = os.path.split(__file__) # path to current folder
    filehead = os.path.dirname(os.path.abspath(__file__))
    print(f'filehead={filehead:s}')

    fig_path = init_directory(filehead, additional_to_name='multiple',additional_folder='/Output_ZSync_UHF_Trigger_Skew')
    daq, scopeModule, wave_nodepath, init_settings = init_ZSync_and_DIO_trigger.init_zsync_UHF_trigger_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER)

    # sort channels_list s.t. correct in calling
    channels_list = []
    for channels_list_i in channels_list_temp:
        channels_list_i.sort()
        channels_list.append(channels_list_i)

    #######################################################
    setting_HDAWGs.set_HDAWG_channels(daq, UHFQA, [0,1], regression)
    # Update the scope
    daq.setInt('/' + str(UHFQA) + '/scopes/0/single', 1)
    ########################################################

    # can comment in or out if you want a preplotting seeing the difference in the UHF scopes
    # preplotting(init_settings, daq, HDAWGS, UHFQA, PQSC, scopeModule, channels_list, wave_nodepath)

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
    s = singlerun_settings

    daq.setInt(f'{PQSC}/execution/enable', 1)
    # Get Trigger progress
    progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')
    scopeModule.execute()
    daq.setInt('/%s/scopes/0/enable' % UHFQA_meas, 1)

    if warmup_time:
        logger.info(f'Now warming up for {warmup_time} minutes.')
        time.sleep(warmup_time*60)

    progress = daq.getDouble(f'/{PQSC}/execution/progress') * 100
    logger.debug(f'{progress:.{2}f} % Trigger Progress')

    data, t_all, results_success = UHF_rpg.multiple_waves(s, daq, HDAWGS, UHFQA, PQSC, scopeModule, channels_list, wave_nodepath, fig_path, number_of_average_runs)

    ZSyncInfo = analysis_helpers.helper_hdawg_get_zsync_dlycal(daq, HDAWGS)

    multiplerun_data = {
        'data'                  : data,
        'ts_all'                : t_all[0],
        'ts_all_error'          : t_all[1],
        'ZSyncInfo'             : ZSyncInfo
        }

    if s['save_raw_data']:
        UHF_pa.save_data(multiplerun_data, s['fig_path'], "raw_data")

    disconnect(daq, PQSC, HDAWGS, UHFQA)

    if not results_success:
        logger.info('No results could be printed.')
        return

    UHF_pa.data_anlysis(multiplerun_data, s)
    logger.info('Results have been printed.')


if __name__ == '__main__':
    '''
    In this main function, a list of the used HDAWGs as well as the channels list containing a list of channels for each HDAWG must be set. The order of the list is thereby the order of the HDAWGS in that list.
    Furthermore, Z-Sync triggering PQSC and Scope-providing UHFQA are input.
    Lastly, the number of runs per acquisition can be set, over which per time shot the average is taken. The minimum number of runs is 2.
    Furthermore, the bool of regression indicates if during the run
    a) a warmup_time given in minutes is initiated
    b) the setup is clocked to a clocking UHF
    c) the specific amplitude ranges for the HDAWGs' output needed for the in-house regression setup
    '''
    ######################### INPUT: #############################

    # The names of all devices in the setup go here:
    #HDAWGS = ['dev8146', 'dev8198', 'dev8246']
    #channels_list_temp = [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3]]

    HDAWGS = ['dev8146', 'dev8246', 'dev8186', 'dev8218']
    channels_list_temp = [[0,1,2,3], [0,1,2,3], [0,1,2,3], [0,1,2,3]]

    #HDAWGS = ['dev8146', 'dev8186', 'dev8246']
    #channels_list_temp = [[0, 1, 2, 3], [0,1,2,3], [0, 1, 2, 3]]

    #HDAWGS = ['dev8146', 'dev8246']
    #channels_list_temp = [[0, 1, 2, 3], [0,1,2,3]]

    #HDAWGS = ['dev8146', 'dev8198']
    #channels_list_temp = [[0, 3, 4, 5, 7],[0, 2, 3, 5, 7]]

    #HDAWGS = ['dev8186', 'dev8198', 'dev8216', 'dev8217']
    #channels_list_temp = [[0, 3, 4, 7], [0, 3, 4, 7], [0, 3, 4, 7], [0,3, 4,7]]

    #HDAWGS = ['dev8146']
    #channels_list_temp = [[0, 1, 2, 3, 4, 5, 6, 7]]

    PQSC = 'dev10021'
    UHFQA_meas = 'dev2333'
    UHF_ext_clock = 'dev2109'

    #PQSC = 'dev10005'
    #UHFQA_meas = 'dev2109'
    #UHF_ext_clock = 'dev2109'

    SERVER = '10.42.0.229'
    SERVER = '10.42.0.228'
    SERVER = 'localhost'

    number_of_average_runs = 10
    warmup_time = 10 # in minutes
    warmup_time = 5
    regression = 2 # 2 is the UHFQA regressioning, 1 is the UHFLI regression
    #regression = 0

    init_experiment = 'simple_trigger'
    #init_experiment = 'feedback_DIO_trig'

    #init_experiment = 'UHF_trigger'
    #channels_list_temp = [[0,1]]

    #print = logger.debug
    ############################################################
    # Run the sequence with all channels on
    ### possible for 'gauss' or 'drag' wave
    ############################################################
    if init_experiment in ['simple_trigger', 'feedback_DIO_trig']:
        run_main_reaction(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, channels_list_temp, number_of_average_runs, regression, warmup_time, init_experiment, SERVER)

    if init_experiment == 'UHF_trigger':
        run_partial_reaction(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, channels_list_temp, number_of_average_runs, regression, warmup_time, init_experiment, SERVER)

    if init_experiment == 'qubit_reset':
        run_qubit_reset(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, channels_list_temp, number_of_average_runs, regression, warmup_time, init_experiment, SERVER)
