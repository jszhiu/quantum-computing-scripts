## -----------------------------------------------------------------------------
## @brief Initialisation Functions for triggering UHFQA test
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Intialisation of clocking UHF, PQSC, HDAWGs and UHF(QA) for the
## triggering UHFQA test, step by step calling in setting_*.py
## ------------------------------------------------------------------------------
import logging
import time

import setting_common
import setting_HDAWGs
import setting_PQSC
import setting_UHF

logger = logging.getLogger()

def init_zsync_UHF_trigger_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER):
    """
    Goal is to have the UHFQA emitting a waveform triggered by the PQSC via DIO of HD and having
    the waveform signal measured by the UHF's scope, which is triggered with the PQSC SMA trigger.
    """

    ######################################
    # 1 ) Initial Settings
    ## a) Connect and initialize devices
    ## b) set the times
    ## d) initialize the scope module
    ######################################
    UHFQA = UHFQA_meas

    daq = setting_common.connection(PQSC, UHFQA, UHF_ext_clock, HDAWGS, regression, SERVER)

    # the times issues:
    #####################################
    #### NOTE: The waveform time length and the scope acquisition time change every time the corresponding sampling rates change because it's given in terms of the sampling rate (so let's just stick to the sampling rates for the time being)
    awg_sampling_rate_variable, awg_sampling_rate_value, awg_waveform_length, scope_sampling_rate, scope_sampling_length, PQSC_trigger_timegap = setting_common.set_times_UHF_trigger()

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
        #pdb.set_trace()

    # Initialize PQSC
    ######################
    PQSC_settings = {
        'PQSC_repetitions'   : 4e9,
        'PQSC_trigger_timegap': PQSC_trigger_timegap,
        'trigger_port'  : 0,
        'DIO_port'      : 0,
        'PQSC'          : PQSC
    }

    setting_PQSC.init_pqsc(daq, PQSC, PQSC_settings)
    setting_PQSC.activate_external_reference_pqsc(daq, PQSC, 1)
    setting_PQSC.receive_and_forward(daq, PQSC, PQSC_settings)

    # Initialize HDAWG
    ########################################
    HDAWG_settings = {
        "awg_sampling_rate_variable" : 0,
        "awg_single_shot": 1,
        "HDAWGS": HDAWGS, #rather alibi here
        'channelgrouping': 0
    }
    setting_HDAWGs.set_external_reference_hdawgs(daq, HDAWGS, 2)
    for i, HDAWG in enumerate(HDAWGS):
        setting_HDAWGs.initialize_awg(daq, HDAWG, '', HDAWG_settings)

    setting_HDAWGs.set_ZSync_and_DIO_feedback(daq, HDAWGS)
    logger.info("Initialization of HDAWGs DONE")

    # initialize UHF
    ##################
    UHFQA_settings = {
        'trigreference'         : 0.1, #0.1, # in percent of how much to the left of trigger should still be in scope length
        'scope_trigholdoff'     : 0.0005, #0.00002, # The scope hold-off time (s), probably the min. time after a trigger, in which no further trigger should be taken (thus certainly shorter than the PQSC Trigger time gap time)
        'scope_length'          : scope_sampling_length,
        'sigouts_amplitude'     : 0.5,
        'scope_time'            : scope_sampling_rate,
        'scope_in_channel'      : 1, # scope input channel --> actually not used in initialization
        'UHFQA'                 : UHFQA,
    }

    waveform = 'drag'
    gap_wait = awg_waveform_length
    AWG_settings = {
        "awg_sampling_rate_variable": awg_sampling_rate_variable,
        "awg_sampling_rate_value":  awg_sampling_rate_value,
        "awg_waveform_length"   : awg_waveform_length,
        "awg_waveform"          : waveform, # gauss or drag
        "gap_wait"              : gap_wait,
        #"awg_source_string_list" : awg_source_string_list,
        "awg_single_shot"       : 1,
        }

    setting_UHF.activate_external_reference_uhf(daq, UHFQA, 1)
    setting_UHF.initialize_uhfqa(daq, UHFQA, Mod, UHFQA_settings)
    #init.DIO_scope_feedback(daq, UHFQA, Mod, UHFQA_settings)
    setting_UHF.set_DIO_to_feedback(daq, UHFQA, UHFQA_settings)
    setting_UHF.DIO_AWG_Trig(daq, UHFQA, AWG_settings)
    #setting_UHF.forward_QA_results_with_DIO(daq, UHFQA, UHFQA_settings)

    wave_nodepath = '/{}/scopes/0/wave'.format(UHFQA)
    #daq.sync()
    logger.info("initialize_uhfqa DONE")

    Mod.subscribe(wave_nodepath)
    init_settings = {**HDAWG_settings, **UHFQA_settings, **AWG_settings, **PQSC_settings}

    time.sleep(5)

    ###################################################
    # Initialisations across devices
    ###################################################
    #Trigger_PQSC_SMA_to_UHF_Ref(daq, PQSC, UHF)
    #ZSync_Connection_PQSC_to_HDAWGs

    return daq, Mod, wave_nodepath, init_settings
