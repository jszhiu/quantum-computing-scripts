## -----------------------------------------------------------------------------
## @brief Helper Functions for Initialisations
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Defining utility functions called by init_*.py:
### a) setter and getter
### b) check_and_set_settings & revision coincidence (not yet called)
### c) get_scope_records and check_scope_record_flags
## 2) HDAWG specific functions (initialization, activating and deactivating
## external reference) Initialize awg_source_strings
## 3) PQSC specific functions (initialization, activating external reference)
## 4) UHF-QA specific functions (initialization, activating external reference)
## ------------------------------------------------------------------------------


'''
Defining utility functions and HDAWG specific functions to be called in Tryout.py:
1) General utility functions (keysetter, keygetter, ooutput directory, check_and_set_settings (UHFQA and HDAWG))
2) HDAWG specific functions (initialization, activating and deactivating external reference) and
3) PQSC specific functions (initialization, activating external reference)
4) UHF-QA specific functions (initialization, activating external reference)
'''
import logging
import zhinst.ziPython as zi

logger = logging.getLogger()

# def gets(self, path):
        # return self.daq.getString(self._get_full_path(path))
# for getting the HDAWG8 or HDAWG4 device type


"""
def gen_output_directory(output_root, additional_to_name='', additional_folder=''):
    ''' Generates Output Directory'''
    fig_id = time.strftime("%Y%m%d-%H%M%S")+additional_to_name
    fig_path = output_root +"/" + fig_id
    if additional_folder != '':
        fig_path = output_root +"/"+ additional_folder + "/"+ fig_id
    # Create new directory.
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    return fig_path"""

"""
def check_and_set_settings(devHD="", devUHF="", serverhost="", port="", hd_interface="", uhf_interface=""): # still to be adjusted to the system used here, not called yet
    ''' Checks for connection setting file and writes a skeletton version of it, if not found.

    If found, returns connection parameters.
    If not found, it checks whether nonempty input paramter has been furnished and ...

    returns connection settings.
    '''
    if devHD == "":
        if not os.path.exists(CONNECTION_SETTINGS_FN):
            connection_skel = {
                "SERVER": "localhost",
                "PORT": 8004,
                "HD_DEVICE": "dev80XX",
                "HD_INTERFACE": "1GbE",
                "UHF_DEVICE": "dev20XX",
                "UHF_INTERFACE": "1GbE"
            }
            with open(CONNECTION_SETTINGS_FN, 'w') as outfile:
                json.dump(connection_skel, outfile, default=ndarray_to_list) # convert ndarrays to lists
            # Alert user.
            assert False, "Please provide connectionsettings in {} dummy file which has been created".format(CONNECTION_SETTINGS_FN)
        else:
            with open(CONNECTION_SETTINGS_FN) as f:
                connection_settings = json.load(f)
            # Check if user has altered dummy
            assert connection_settings["HD_DEVICE"] != "dev80XX"  and connection_settings["UHF_DEVICE"] != "dev20XX",\
            "Please input valid connection settings in {} dummy file".format(CONNECTION_SETTINGS_FN)
    else:
        connection_settings = {
            "SERVER": serverhost,
            "PORT": port,
            "HD_DEVICE": devHD,
            "HD_INTERFACE": hd_interface,
            "UHF_DEVICE": devUHF,
            "UHF_INTERFACE": uhf_interface
        }

    return connection_settings"""

"""
def check_revision_coincidence(UHF, HD):# Get information about dataserver and API revision.   --> Are these really necessary?, not called yet
    ''' Check if DAQ revision number and dataserver revision numbers coincide '''
    dataserver_revision = UHF.geti('/zi/about/revision')
    hd_daq_revision = HD.daq.revision()
    uhf_daq_revision = UHF.daq.revision()
    print("daq.getInt('/zi/about/revision'):{}.".format(dataserver_revision))  # Data Server revision
    print("UHF daq.revision(): {}.".format(uhf_daq_revision)) # Python API revision
    print("HD daq.revision(): {}.".format(hd_daq_revision)) # Python API revision

    #assert dataserver_revision == uhf_daq_revision and dataserver_revision == hd_daq_revision,\
         #"Version Mismatch Dataserver - UHF DAQ or HD DAQ"
"""

def connection(PQSC, UHFQA, UHF_ext_clock, HDAWGS, regression, SERVER):
    """
    # 3) Initialize Devices and Modules
    ## a) Define the daq
    ## b) Initialize 1 PQSC, the needed HDAWGs and 1 UHF
    ## c) Initialize the used AWG Module of the HDAWGs
    """

    PORT = 8004
    INTERFACE = '1GbE'

    # Server connection
    daq = zi.ziDAQServer(SERVER, PORT, 5)
    daq.setDebugLevel(0)

    # Disable everything that is connected at the time being from a potential previous run
    # TODO: Implement disable_everything for all but the clocking UHF
    """
    x = daq.getByte('/zi/devices/connected')
    ConnectedDevices = x.split(",")
    for device in ConnectedDevices:
        dev_type = gets(daq, device, 'features/devtype')
        if dev_type == ('HDAWG8' or 'HDAWG4') and (device not in HDAWGS):
            utils.disable_everything(daq, device)
    """

    if regression:
        daq.connectDevice(UHF_ext_clock, INTERFACE)
        logger.info(f'Done connecting to external clock UHF {UHF_ext_clock}')

    # instrument connection
    for HDAWG in HDAWGS:
        daq.connectDevice(HDAWG, INTERFACE)
        logger.info(f'Done connecting to {HDAWG}')
        #seti(daq, HDAWG, 'system/clocks/referenceclock/source', 0)
        daq.setInt(f'{HDAWG}/system/clocks/referenceclock/source', 0)
        #daq.sync()
        logger.info(f'{HDAWG} clock set to internal')

    daq.connectDevice(PQSC, INTERFACE)
    logger.info(f'Done connecting to {PQSC}')
    daq.setInt(f'{PQSC}/system/clocks/referenceclock/in/source', 0)
    #seti(daq, PQSC, 'system/clocks/referenceclock/in/source', 0)
    #daq.sync()
    logger.info(f'{PQSC} clock set to internal')

    daq.connectDevice(UHFQA, INTERFACE)
    logger.info(f'Done connecting to measuring UHF {UHFQA}')
    #activate_external_reference_uhf(daq, UHFQA, 0)
    #seti(daq, UHFQA, 'system/extclk', 0)
    daq.setInt(f'{UHFQA}/system/extclk', 0)
    #daq.sync()
    logger.info(f'{UHFQA} clock set to internal')
    """
    daq.connectDevice(UHF_ext_clock, INTERFACE)
    logger.info(f'Done connecting to external clock UHF {UHF_ext_clock}')"""

    # If running: disable the PQSC trigger execution
    #seti(daq, PQSC, 'execution/enable', 0)
    daq.setInt(f'{PQSC}/execution/enable', 0)
    #daq.sync()

    return daq

def get_HDAWG_sampling_rate(key):
    HDAWG_sampling_rate = {
        0: 2.4e9,
        }
    return HDAWG_sampling_rate[key]

def get_UHF_sampling_rate(key):
    UHF_sampling_rate = {
        0: 1.8e9,
        }
    return UHF_sampling_rate[key]

def set_times_trigger():
    # the times issues:
    #####################################
    #### NOTE: The waveform time length and the scope acquisition time change every time the corresponding sampling rates change because it's given in terms of the sampling rate (so let's just stick to the sampling rates for the time being)
    awg_sampling_rate_variable = 0 # awg_sampling_rate_variable : [0, 1, ..., 13] for [2.4 GHz, 1.2 GHz, ..., 292.86 kHz]
    awg_sampling_rate_value = get_HDAWG_sampling_rate(awg_sampling_rate_variable)
    awg_waveform_length = 2**5 # smallest possible waveform length of 32

    logger.info('Period per output pulse roughly {} ns'.format(0.42*awg_waveform_length)) # the 0.42 comes from the fact that 1/2.4e9 = 0.42 ns

    scope_sampling_rate = 0 # scope_sampling_rate : [0, 1, .., 16] for [1.8 GHz, 900 MHz, ..., 27.5 kHz]
    # note that sampling_length is always given in number of samples (can range from 2**12 to 2**27)
    scope_sampling_length = 2**12 # to be above 2 us

    logger.info('Period per UHF acquisition shot roughly {} ns'.format(0.5555*scope_sampling_length)) # the 0.5555 comes from the fact that 1/1.8e9 = 0.5555 ns

    PQSC_trigger_timegap = 6e-6 # Should be in any case a multiple of both 300 MHz and 450 (?) MHz (sequencer of both HD and UHF) and larger than the scope time
    logger.info('Period per PQSC trigger roughly {} ns'.format(PQSC_trigger_timegap*1e9))

    return awg_sampling_rate_variable, awg_sampling_rate_value, awg_waveform_length, scope_sampling_rate, scope_sampling_length, PQSC_trigger_timegap

def set_times_feedback():
    # the times issues:
    #####################################
    #### NOTE: The waveform time length and the scope acquisition time change every time the corresponding sampling rates change because it's given in terms of the sampling rate (so let's just stick to the sampling rates for the time being)

    awg_sampling_rate_variable = 0 # awg_sampling_rate_variable : [0, 1, ..., 13] for [2.4 GHz, 1.2 GHz, ..., 292.86 kHz] in case of HDAWG and [1.8 GHz, ...] in case of UHFQA
    awg_sampling_rate_value = get_HDAWG_sampling_rate(awg_sampling_rate_variable)
    awg_waveform_length = 2**5 # smallest possible waveform length of 32

    logger.info('Period per output pulse roughly {} ns'.format(0.42*awg_waveform_length)) # the 0.42 comes from the fact that 1/2.4e9 = 0.42 ns

    scope_sampling_rate = 0 # scope_sampling_rate : [0, 1, .., 16] for [1.8 GHz, 900 MHz, ..., 27.5 kHz]
    # note that sampling_length is always given in number of samples (can range from 2**12 to 2**27)
    scope_sampling_length = 2**12 # to be above 2 us

    logger.info('Period per UHF acquisition shot roughly {} ns'.format(0.5555*scope_sampling_length)) # the 0.5555 comes from the fact that 1/1.8e9 = 0.5555 ns

    PQSC_trigger_timegap = 2
    PQSC_trigger_timegap = 6.2e-6 # Should be in any case a multiple of both 300 MHz and 450 MHz (sequencer of both HD and UHF) and larger than the scope time
    logger.info('Period per PQSC trigger roughly {} ns'.format(PQSC_trigger_timegap*1e9))

    return awg_sampling_rate_variable, awg_sampling_rate_value, awg_waveform_length, scope_sampling_rate, scope_sampling_length, PQSC_trigger_timegap

def set_times_UHF_trigger():
    # the times issues:
    #####################################
    #### NOTE: The waveform time length and the scope acquisition time change every time the corresponding sampling rates change because it's given in terms of the sampling rate (so let's just stick to the sampling rates for the time being)

    awg_sampling_rate_variable = 0 # awg_sampling_rate_variable : [0, 1, ..., 13] for [1.8 GHz, ...]
    awg_sampling_rate_value = get_UHF_sampling_rate(awg_sampling_rate_variable)

    awg_waveform_length = 2**5 # smallest possible waveform length of 32

    logger.info('Period per output pulse roughly {} ns'.format(0.42*awg_waveform_length)) # the 0.42 comes from the fact that 1/2.4e9 = 0.42 ns

    scope_sampling_rate = 0 # scope_sampling_rate : [0, 1, .., 16] for [1.8 GHz, 900 MHz, ..., 27.5 kHz]
    # note that sampling_length is always given in number of samples (can range from 2**12 to 2**27)
    scope_sampling_length = 2**12 # to be above 2 us

    logger.info('Period per UHF acquisition shot roughly {} ns'.format(0.5555*scope_sampling_length)) # the 0.5555 comes from the fact that 1/1.8e9 = 0.5555 ns

    PQSC_trigger_timegap = 6e-6 # Should be in any case a multiple of both 300 MHz and 450 (?) MHz (sequencer of both HD and UHF) and larger than the scope time
    logger.info('Period per PQSC trigger roughly {} ns'.format(PQSC_trigger_timegap*1e9))

    return awg_sampling_rate_variable, awg_sampling_rate_value, awg_waveform_length, scope_sampling_rate, scope_sampling_length, PQSC_trigger_timegap
