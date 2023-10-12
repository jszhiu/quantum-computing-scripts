## -----------------------------------------------------------------------------
## @brief Initialisation Functions for triggering sync test
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Intialisation of clocking UHF, PQSC, HDAWGs and UHF(QA) for the
## triggering sync test, step by step calling in setting_*.py
## ------------------------------------------------------------------------------

import logging

import setting_common
import setting_HDAWGs
import setting_PQSC
import setting_UHF

logger = logging.getLogger()

def init_zsync_trigger_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER):
    '''
    Initialize the test of checking waveform signal 'gauss' or 'drag' exiting from the list of HDAWGs (not yet set out of which channels) ZSync triggered by the PQSC and connected to the UHF acting as a scope.

    Input:  PQSC : device name
            HDAWGS: list containing HDAWG device names
            UHFQA: device name

    Ouput:  daq: initialized data acquisition run
            scopeModule: scope Module initialized on UHFQA
            wave_nodepath: wave nodepath of scope. Subscribed to it.
            init_settings: dictionary containing the settings to all devices
            fig_path: generated path to output directory where figures and json files genereted in the course of the experiment should be saved to

    The Function is set up as follows:
    1) Setting initial settings (connection, fig_path generation, setting scope rates)
    2) Defining pulse sequence sourcestrings
    3) Define daq, connect to devices, initialize scopeModule
    4) Set the PQSC, HDAWGs and UHF settings
    Main experiment parameters are set in this function in 4) and have to be adjusted within this function.
    '''

    ######################################
    # 1 ) Initial Settings
    ## a) Connect and initialize devices
    ## b) set the times
    ## c) Define pulse sequence source strings
    ## d) initialize the scope module
    ######################################
    UHFQA = UHFQA_meas

    daq = setting_common.connection(PQSC, UHFQA, UHF_ext_clock, HDAWGS, regression, SERVER)

    # the times issues:
    #####################################
    #### NOTE: The waveform time length and the scope acquisition time change every time the corresponding sampling rates change because it's given in terms of the sampling rate (so let's just stick to the sampling rates for the time being)
    awg_sampling_rate_variable, awg_sampling_rate_value, awg_waveform_length, scope_sampling_rate, scope_sampling_length, PQSC_trigger_timegap = setting_common.set_times_trigger()

    # Define the Pulse sequence sourcestrings
    #######################################

    # The own Gaussian or Drag Wave
    gap_wait = awg_waveform_length # minimum possible gap for a wave
    waveform = 'drag'
    awg_source_string_list = setting_HDAWGs.make_sourcestring_trigger(awg_waveform_length, gap_wait, waveform, len(HDAWGS))

    # Initialize scope module
    Mod = setting_UHF.init_uhf_scope(daq)

    #####################################
    # 2) Initilize Experiment
    ## a) Set the external UHF clock triggering the PQSC
    ## b) Set the Master PQSC
    ## c) Set the chosen HDAWGs
    ## d) Set the UHF and subscribe to scopeModule
    #####################################
    # Initialize the UHF generating the clock
    #####################################
    if regression:
        setting_UHF.ext_clock_on(daq, UHF_ext_clock)
        logger.info(f'Done setting external clock on {UHF_ext_clock}')

    # Initialize PQSC
    ######################
    PQSC_settings = {
        'PQSC_repetitions'      : 4e9,
        'PQSC_trigger_timegap'  : PQSC_trigger_timegap,
        'trigger_port'          : 0,
        'DIO_port'              : 0,
        'PQSC'                  : PQSC
    }

    setting_PQSC.init_pqsc(daq, PQSC, PQSC_settings)
    if regression:
        setting_PQSC.activate_external_reference_pqsc(daq, PQSC, 1)

    # Initialize HDAWG
    ########################################
    HDAWGS_channel_types = []
    HDAWGS_delay_types = []
    for device in HDAWGS:
        dev_type = daq.getString(f'/{device}/features/devtype')
        HDAWGS_channel_types.append(dev_type)
        options = daq.getString(f'/{device}/features/options')
        split_options = options.split("\n")
        if 'SKW' in split_options:
            HDAWGS_delay_types.append('SKW')
        else:
            HDAWGS_delay_types.append('No_SKW')

    HDAWG_settings = {
        # Select channel grouping awgsXchannels varying in 4X2 (0), 2X4 (1), 1X8 (2)
        "channelgrouping"       : 0,
        # Set sampling rate to (highest) 2.4 GHz
        "awg_sampling_rate_variable": awg_sampling_rate_variable,
        "awg_sampling_rate_value":  awg_sampling_rate_value,
        "awg_waveform_length"   : awg_waveform_length,
        "awg_waveform"          : waveform, # gauss or drag
        "HDAWGS"                : HDAWGS,
        "HDAWGS_channel_types"  : HDAWGS_channel_types,
        #"HDAWGS_delay_types"    : HDAWGS_delay_types,
        "gap_wait"              : gap_wait,
        "awg_source_string_list" : awg_source_string_list,
        "awg_single_shot"       : 1
    }

    for i, HDAWG in enumerate(HDAWGS):
        setting_HDAWGs.initialize_awg(daq, HDAWG, awg_source_string_list[i], HDAWG_settings)
        # Set DIO to ZSync Triggering
        setting_HDAWGs.set_dio_mode_zsync(daq, HDAWG)
    logger.info("Initialization of HDAWGs DONE")

    # HDAWGs lock reference clock to ZSYNC coming from PQSC
    setting_HDAWGs.set_external_reference_hdawgs(daq, HDAWGS, 2)

    # initialize UHF
    ##################
    UHFQA_settings = {
        'trigreference'         : 0.1, #0.1, # in percent of how much to the left of trigger should still be in scope length
        'scope_trigholdoff'     : 0.0005, #0.00002, # The scope hold-off time (s), probably the min. time after a trigger, in which no further trigger should be taken (thus certainly shorter than the PQSC Trigger time gap time)
        'scope_length'          : scope_sampling_length,
        'sigouts_amplitude'     : 0.5,
        'scope_time'            : scope_sampling_rate,
        'scope_in_channel'      : 1, # scope input channel --> actually not used in initialization
        'UHFQA'                 : UHFQA
    }

    if regression:
        setting_UHF.activate_external_reference_uhf(daq, UHFQA, 1)
    setting_UHF.initialize_uhfqa(daq, UHFQA, Mod, UHFQA_settings)
    setting_UHF.frontpanel_scope_trig(daq, UHFQA, Mod, UHFQA_settings)

    wave_nodepath = '/{}/scopes/0/wave'.format(UHFQA)
    #daq.sync()
    logger.info("initialize_uhfqa DONE")

    Mod.subscribe(wave_nodepath)
    init_settings = {**HDAWG_settings, **UHFQA_settings, **PQSC_settings}

    ###################################################
    # Initialisations across devices
    ###################################################
    #Trigger_PQSC_SMA_to_UHF_Ref(daq, PQSC, UHF)
    #ZSync_Connection_PQSC_to_HDAWGs

    return daq, Mod, wave_nodepath, init_settings
