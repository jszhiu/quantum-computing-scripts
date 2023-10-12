## -----------------------------------------------------------------------------
## @brief Scope results of trigger and feedback test
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Only one run of the initialised test, separated by channel inputs
## intended to be used for debugging
## ------------------------------------------------------------------------------

import time
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import setting_functions as init
import init_and_disconnect_skew_msrmt as initdisc

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def preplotting(init_settings, daq, HDAWGS, UHF, PQSC, scopeModule, channels_list_temp, wave_nodepath, regression):
    '''
    Running the already initialized devices and getting the plot of the measured signal on each of the UHF scope inputs.

    Input:  init_settings   : dictionary (needed to loop over the scope channels)
            daq             : current data acquisition run
            HDAWGS          : list of device names of certain order
            channels_list   : list of channels corresponding to the to-be enabled output channels of the HDAWGs in the HDAWGS list. Note that the order must be the same as in that list
            UHF             : device name
            PQSC            : device name
            scopeModule     : already initialized scope module

    Output: None
    '''

    # sort channels_list s.t. correct in calling
    channels_list = []
    for channels_list_i in channels_list_temp:
        channels_list_i.sort()
        channels_list.append(channels_list_i)

    for i, HDAWG in enumerate(HDAWGS):
        init.set_HDAWG_channels(daq, HDAWG, channels_list[i], regression)

    # Update the scope
    daq.setInt('/' + str(UHF) + '/scopes/0/single', 1)

    daq.sync()

    init.seti(daq, PQSC, 'execution/enable', 1)
    # Get Trigger progress
    # progress = init.getd(daq, PQSC, 'execution/progress') * 100
    # logger.info(f'{progress:.{2}f} % Trigger Progress in preplot')

    # Enable the scope and read the scope data arriving from the device. Note: The module is already configured and the
    # required data is already subscribed
    data = get_scope_records(UHF, daq, scopeModule)
    print (data)
    loop = data
    for node in loop:
        print (node)

    assert wave_nodepath in data, "The Scope Module did not return data for {}. Check the UHF trigger input.".format(wave_nodepath)
    logger.debug('Number of scope records returned with triggering enabled: {}.'.format(len(data[wave_nodepath])))
    check_scope_record_flags(data[wave_nodepath])

    scope_in_channel = init_settings['scope_in_channel']
    scope_time = init_settings['scope_time']

    for scope_in_channel in [0,1]:
        plot_scope(daq, UHF, data[wave_nodepath], scope_in_channel, scope_time)

def get_scope_records(device, daq, scopeModule):
    """
    Obtain scope records from the device using an instance of the Scope Module.
    """

    # Tell the module to be ready to acquire data; reset the module's progress to 0.0.
    scopeModule.execute()

    # Enable the scope: Now the scope is ready to record data upon receiving triggers.
    daq.setInt('/%s/scopes/0/enable' % device, 1)
    # daq.setDouble('/%sscopes/0/segments/count' % device, n_segments)

    daq.sync()

    start = time.time()
    timeout = 50  # [s]
    records = 0
    progress = 0
    # Wait until the Scope Module has received and processed the desired number of records.
    while progress < 1.0:
        time.sleep(0.5)
        records = scopeModule.getInt("records")
        progress = scopeModule.progress()[0]
        if (time.time() - start) > timeout:
            # Break out of the loop if for some reason we're no longer receiving scope data from the device.
            logger.debug("\nScope Module did not return records after {} s - forcing stop.".format(timeout))
            break
    print("")
    daq.setInt('/%s/scopes/0/enable' % device, 0)

    # Read out the scope data from the module.
    data = scopeModule.read(True)
    help(scopeModule.read)

    # disable scope --> check if it works
    #zi_shell_device_uhf.seti('scopes/0/enable', 0)

    # Stop the module; to use it again we need to call execute()
    scopeModule.finish()

    return data

def check_scope_record_flags(scope_records):
    """
    Loop over all records and print a warning to the console if an error bit in
    flags has been set.

    Warning: This function is intended as a helper function for the API's
    examples and it's signature or implementation may change in future releases.
    """
    num_records = len(scope_records)
    #pdb.set_trace()
    for index, record in enumerate(scope_records):
        if record[0]['flags'] & 1:
            logger.warning('Warning: Scope record {}/{} flag indicates dataloss.'.format(index, num_records))
        if record[0]['flags'] & 2:
            logger.warning('Warning: Scope record {}/{} indicates missed trigger.'.format(index, num_records))
        if record[0]['flags'] & 4:
            logger.warning('Warning: Scope record {}/{} indicates transfer failure (corrupt data).'.format(index, num_records))
        totalsamples = record[0]['totalsamples']
        for wave in record[0]['wave']:
            # Check that the wave in each scope channel contains the expected number of samples.
            assert len(wave) == totalsamples, \
                'Scope record {}/{} size does not match totalsamples.'.format(index, num_records)

def plot_scope(daq, device, scope_records, scope_in_channel, scope_time=0):
    '''
    Plot the signal coming into the scope in scope_in_channel. Potentially several scope records possible and all of them plotted if so.
    '''
    # Get the instrument's ADC sampling rate.
    clockbase = daq.getInt('/{}/clockbase'.format(device)) # equals 1.8 GHz
    for index, record in enumerate(scope_records):
        record = record[0]
        print('Scope Record Nodes: ')
        for node in record:
            print (node, record[node])
        print ('Scope Record Header Nodes: ')
        loop = record['header']
        for node in loop:
            print (node, loop[node])
        # pdb.set_trace()
        totalsamples = record['totalsamples']
        wave = record['wave'][scope_in_channel, :]
        if record['flags'] & 7:
            logger.debug(f"Skipping plot of record {index}: record flags= {record['flags']} indicate corrupt data.")
            continue
        if not record['channelmath'][scope_in_channel] & 2:
            # We're in time mode: Create a time array relative to the trigger time.
            dt = record['dt']
            # The timestamp is the timestamp of the last sample in the scope segment.
            timestamp = record['timestamp']
            triggertimestamp = record['triggertimestamp']
            t = np.arange(-totalsamples, 0)*dt + (timestamp - triggertimestamp)/float(clockbase)
            plt.plot(1e6*t, wave)
        elif record['channelmath'][scope_in_channel] & 2:
            # We're in FFT mode.
            scope_rate = clockbase/2**scope_time
            f = np.linspace(0, scope_rate/2, totalsamples)
            plt.semilogy(f/1e6, wave)

    plt.draw()
    plt.grid(True)
    plt.ylabel('Amplitude [V]')
    plt.autoscale(enable=True, axis='x', tight=True)

    plt.axvline(0.0, linewidth=2, linestyle='--', color='black', label="Trigger time")
    plt.title('{} Scope records from {} in UHF channel {}'.format(len(scope_records), device, scope_in_channel))
    plt.xlabel('t (relative to trigger) [us]')
    plt.legend()
    #plt.show()
    plt.close('all')


if __name__ == '__main__':
    '''
    In tis main function, a list of the used HDAWGs as well as the channels list containing a list of channels for each HDAWG must be set. The order of the list is thereby the order of the HDAWGS in that list.
    Furthermore, Z-Sync triggering PQSC and  Scope-providing UHFQA are input, and the waveform = 'drag' or waveform = 'gauss'.
    Lastly, the number of runs per acquisition can be set, over which per time shot the average is taken.
    '''

    ######################### INPUT: #############################
    # The names of all devices in the setup go here:
    HDAWGS = [ 'dev8146', 'dev8198', 'dev8143']
    channels_list_temp = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]


    #HDAWGS = ['dev8146', 'dev8143']
    #channels_list_temp = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]

    HDAWGS = [ 'dev8146']
    channels_list_temp = [[0, 1, 2, 3, 4, 5, 6, 7]]

    PQSC = 'dev10004'
    UHFQA_meas = 'dev2333'
    UHF_ext_clock = 'dev2107'

    regression = True

    init_experiment = 'simple_trigger'
    init_experiment = 'feedback'
    #init_experiment = 'feedback_DIO_trig'
    init_experiment = 'UHF_trigger'

    """
    HDAWGS = ['dev8035','dev8015']
    channels_list_temp = [[0, 1, 2, 3, 4, 5, 6, 7],[0, 1, 2, 3]]

    PQSC = 'dev10004'
    UHFQA_meas = 'dev2006'
    UHF_ext_clock = 'dev2048'
    regression = True
    experiment = 'simple_trigger'
    """

    SERVER = '10.42.0.228'

    number_of_average_runs = 10
    ################################################################
    # Generate output directory.
    """
    filehead, _ = os.path.split(__file__) # path to current folder

    if init_experiment == 'feedback':
        fig_path = initdisc.init_directory(filehead, additional_folder='/Output_ZSync_Feedback_Skew')
        daq, scopeModule, wave_nodepath, init_settings = initdisc.init_zsync_feedback_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER)

    if init_experiment == 'simple_trigger':
        fig_path = initdisc.init_directory(filehead, additional_folder='/Output_ZSync_Trigger_Skew')
        daq, scopeModule, wave_nodepath, init_settings = initdisc.init_zsync_trigger_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER)

    if init_experiment == 'feedback_DIO_trig':
        fig_path = initdisc.init_directory(filehead, additional_folder='/Output_ZSync_FeedbackDIO_Skew')
        daq, scopeModule, wave_nodepath, init_settings = initdisc.init_zsync_feedbackDIO_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER)

    if init_experiment == 'UHF_trigger':
        fig_path = initdisc.init_directory(filehead, additional_to_name='multiple',additional_folder='/Output_ZSync_UHF_Trigger_Skew')
        daq, scopeModule, wave_nodepath, init_settings = initdisc.init_zsync_UHF_trigger_experiment(PQSC, UHFQA_meas, UHF_ext_clock, HDAWGS, regression, SERVER)

    preplotting(init_settings, daq, HDAWGS, UHFQA_meas, PQSC, scopeModule, channels_list_temp, wave_nodepath, regression)
    initdisc.disconnect(daq, PQSC, HDAWGS, UHFQA_meas"""
