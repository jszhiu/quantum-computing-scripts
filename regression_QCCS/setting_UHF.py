## -----------------------------------------------------------------------------
## @brief Helper Functions for Initialisations
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Defining UHF specific utility functionscalled by init_*.py:
### a) ext_clock_on(daq, UHF_ext_clock)
### b) initialize_uhfqa(daq, device, scopeModule, settings)
### c) activate_external_reference_uhf(daq, device, source)
### d) init_uhf_scope(daq)
### e) Scope Triggering: frontpanel_scope_trig, backpanel_scope_trig,
###### DIO_scope_feedback(daq, UHFQA, Mod, UHFQA_settings)
### f) Feedback specific: generate_QA_results, set_DIO_to_feedback
### g) DIO_AWG_Trig(daq, UHFQA, AWG_settings)
## ------------------------------------------------------------------------------
import logging
import time

from time import sleep

logger = logging.getLogger()

############################################
## 4) UHF-QA specific functions
### a) define class ResultLoggingSource(enum.IntEnum)
### b) initialize_uhfqa(daq, device)
################################

def ext_clock_on(daq, UHF_ext_clock):
    """
    Generating the external clock for Ref/Trig 1 leading to the PQSC and for SigOut 1 to the
    measuring UHF in LI case // Ref/Trig 2 to the measuring UHF in QA case
    """
    uhf_osc_freq = 10e6
    assert uhf_osc_freq in (10e6, 100e6), 'Demodulator does not know how to handle ref clock frequency'

    # Signal inputs
    daq.setDouble(f'/{UHF_ext_clock}/sigins/0/range', 1.5)
    daq.setInt(f'/{UHF_ext_clock}/sigins/0/imp50', 1)

    daq.setDouble(f'/{UHF_ext_clock}/sigins/1/range', 1.5)
    daq.setInt(f'/{UHF_ext_clock}/sigins/1/imp50', 1)

    # Demodulators
    daq.setInt(f'/{UHF_ext_clock}/demods/*/enable', 0)
    daq.setInt(f'/{UHF_ext_clock}/demods/0/oscselect', 0)
    daq.setInt(f'/{UHF_ext_clock}/demods/0/adcselect', 0)
    daq.setInt(f'/{UHF_ext_clock}/demods/0/enable', 1)

    daq.setInt(f'/{UHF_ext_clock}/demods/0/order', 8)
    daq.setDouble(f'/{UHF_ext_clock}/demods/0/timeconstant', 0.000366291684)

    # Signal outputs
    #daq.setInt(f'/{UHF_ext_clock}/sigouts/0/imp50', 1)
    #daq.setInt(f'/{UHF_ext_clock}/sigouts/1/imp50', 1)
    #daq.setInt(f'/{UHF_ext_clock}/sigouts/0/range', 1)
    #daq.setInt(f'/{UHF_ext_clock}/sigouts/1/range', 1)
    daq.setInt(f'/{UHF_ext_clock}/sigouts/0/enables/*', 0)
    daq.setInt(f'/{UHF_ext_clock}/sigouts/1/enables/*', 0)
    daq.setInt(f'/{UHF_ext_clock}/sigouts/0/enables/0', 1)
    daq.setInt(f'/{UHF_ext_clock}/sigouts/1/enables/0', 1)
    daq.setDouble(f'/{UHF_ext_clock}/sigouts/0/amplitudes/0', 0.75) # V_peak
    daq.setDouble(f'/{UHF_ext_clock}/sigouts/1/amplitudes/0', 0.75)
    daq.setInt(f'/{UHF_ext_clock}/sigouts/0/on', 1)
    daq.setInt(f'/{UHF_ext_clock}/sigouts/1/on', 1)

    # Oscillator frequency will be set dynamically during the test
    #daq.setDouble(f'/{UHF_ext_clock}/oscs/0/freq', uhf_osc_freq)

    daq.setInt(f'/{UHF_ext_clock}/triggers/out/0/source', 1)
    daq.setInt(f'/{UHF_ext_clock}/triggers/out/0/drive', 1)

    daq.setInt(f'/{UHF_ext_clock}/triggers/out/1/source', 1)
    daq.setInt(f'/{UHF_ext_clock}/triggers/out/1/drive', 1)

    daq.setDouble(f'/{UHF_ext_clock}/oscs/0/freq', uhf_osc_freq)
    #daq.sync()
    sleep(5)

def initialize_uhfqa(daq, device, scopeModule, settings):
    '''
    Initializing a UHF device settings.

    Input:  daq: current run in data aquisition
            device: UHF device name
            scopeModule: already initialized scopeModule
            settings: dictionary containing how the UHF settings should be initialized

    Output: None
    '''

    s = settings
    scope_length = s['scope_length']
    sigouts_amplitude = s['sigouts_amplitude']
    scope_time = s['scope_time']

    parameters = [
        # BASIC INPUT CHANNEL SETTINGS
        # Input ranges
        ('sigins/*/range', 4),
        ('sigins/*/imp50', 1),
        ('sigins/*/ac', 0),
        # SCOPE SETTINGS (from Can)
        # Set sampling rate to (highest) 1.8 GHz:
        ('scopes/0/time', scope_time),
        ('scopes/0/single', 1),
        # from the tutorial
        ('scopes/0/length', scope_length),
        ('scopes/0/channel',  3), # activate both channels
        # Choosing between sample averaging (0) and decimation (1)
        ('scopes/0/channels/*/bwlimit', 1),

        # Select input: 0=Signal input 1 / 1=Signal input 2 /// and the channel infront to which channel it is linked to
        ('scopes/0/channels/0/inputselect', 0),
        ('scopes/0/channels/1/inputselect', 1),
        # 'segments/enable' : Disable segmented data recording.
        ('scopes/0/segments/enable', 0),
        ]

    for node, key in parameters:
        daq.setInt(f'{device}/{node}', key)

def activate_external_reference_uhf(daq, device, source):
    '''
    Setting the PQSC clock to the external reference by setting source = 1.
    Setting the PQSC clock
    '''
    logger.info("Set reference-clock to external")
    daq.setInt(f'{device}/system/extclk', source)

    # wait until both status are 0 and the sources correct,
    # or timeout occurs:
    """poll_cnt = 0
    while True:
        stat = 0
        ok = True
        stat = geti(daq, device, 'system/extclk')
        src  = geti(daq, device, 'system/extclk')
        if stat != 0 or src != source:
            ok = False

        # all stat need to be 0, all src need to be source
        if ok:
            print ("Done")
            return True
        poll_cnt +=1
        if poll_cnt > 100:
            assert poll_cnt > 100, "Timeout, failed!"
            """
    time.sleep(15)
    #print(str(poll_cnt), end=' ')
    #time.sleep(10)
    #daq.sync()

def init_uhf_scope(daq):
    module_averager_weight = 1
    Mod = daq.scopeModule()
    Mod.set('mode', 1) # Time mode = 1, Frequency mode = 3
    Mod.set('averager/weight', module_averager_weight)
    assert Mod.get('scopeModule/averager/weight')['averager']['weight'][0] == module_averager_weight,\
        "Scope weight set unsuccessfull"

    Mod.set('scopeModule/clearhistory', 1)
    logger.info('Initializing scope module of UHFQA DONE')
    return Mod

def frontpanel_scope_trig(daq, UHFQA, Mod, UHFQA_settings):
    s = UHFQA_settings
    device = UHFQA
    trigreference = s['trigreference']
    scope_trigholdoff = s['scope_trigholdoff']

    parameters = [
        ('scopes/0/trigchannel', 2), # Trigchannel 2 is the Trigger 1 input
        ('scopes/0/trigslope', 1),
        ('scopes/0/trigenable', 1),
        ]

    dparameters = [
        ('scopes/0/trigholdoff', scope_trigholdoff),
        ('scopes/0/trigreference', trigreference),
        ('scopes/0/triglevel', 200)
        ]

    for node, key in parameters:
        daq.setInt(f'{device}/{node}', key)

    for node, key in dparameters:
        daq.setDouble(f'{device}/{node}', key)

def backpanel_scope_trig(daq, UHFQA, Mod, UHFQA_settings):
    """
    The scope can not be directly triggered by the Trigger in the back. Therefore, the backpanel trigger triggers the AWG, which then triggers the scope.
    """
    device = UHFQA
    s = UHFQA_settings
    ###################################################
    # a) Backpanel trigger in triggering UHF AWG
    ###################################################
    #daq.setInt('/dev2333/awgs/0/triggers/0/gate/inputselect', 0)
    daq.setInt(f'{device}/awgs/0/auxtriggers/0/channel', 2)
    daq.setInt(f'{device}/awgs/0/auxtriggers/0/slope', 1)
    daq.setInt(f'{device}/awgs/0/single', 1)

    ##################################################
    # b) UHF AWG taking trigger and running the sequence
    ##################################################

    uhf_awg_source_string = """
    while (1) {
        waitDigTrigger(1, 1);
        setTrigger(1);
        // setUserReg(0, 1);
        setTrigger(0);
        }"""

    ####################################
    # Upload waveform onto UHF
    ###################################
    print ('starting to upload triggering waveform onto UHF AWG')

    h = daq.awgModule()
    h.set('awgModule/device', UHFQA)
    #h.set('awgModule/index', 1)
    h.execute()

    h.set('awgModule/compiler/sourcestring', uhf_awg_source_string)

    # Start Compiler
    while h.getInt('awgModule/compiler/status') == -1:
        time.sleep(0.1)
        print('.', end='')
    if h.getInt('awgModule/compiler/status') == 0:
        print("Compiler successful!")
    else:
        raise Exception("Compiler ERROR: "+str(h.getInt('awgModule/compiler/status')) + " " +str(h.getString('awgModule/compiler/statusstring')))

    # Upload ELF
    h.set('awgModule/elf/upload', 1)
    while h.getInt('awgModule/elf/upload') == 1:
        time.sleep(0.1)
        print('.', end='')
    elfstatus = h.getInt('awgModule/elf/status')
    if elfstatus == 0:
        print("ELF upload successful!")
    else:
        raise Exception("ELF upload ERROR! " + str(elfstatus))

    daq.setInt(f'{device}/awgs/0/enable', 1)

    ##############################################################
    # c) setting scope trigger on waiting for AWG trigger
    #############################################################

    trigreference = s['trigreference']
    scope_trigholdoff = s['scope_trigholdoff']

    parameters = [
        # Triggering
        ('scopes/0/trigchannel', 192), # set the make scope recieving AWG 1 Trigger
        ('scopes/0/trigslope', 1),
        ('scopes/0/trigenable', 1),
        ]

    dparameters = [
        ('scopes/0/trigholdoff', scope_trigholdoff),
        ('scopes/0/trigreference', trigreference),
        ('scopes/0/triglevel', 200)
        ]

    for node, key in parameters:
        daq.setInt(f'{device}/{node}', key)

    for node, key in dparameters:
        daq.setDouble(f'{device}/{node}', key)

def DIO_scope_feedback(daq, UHFQA, Mod, UHFQA_settings):
    """
    The scope can not be directly triggered by the DIO Trigger. Therefore, the DIO trigger triggers the AWG, which then first triggers the fake qubit measurement and result of an excited state and then triggers the scope after some wait time.
    """
    device = UHFQA
    s = UHFQA_settings
    ###################################################
    # a) DIO trigger in triggering UHF AWG
    ###################################################
    #seti(daq, device, 'awgs/0/dio/strobe/slope', 0) # Todo: maybe put to function forward_QA_results_with_DIO
    #seti(daq, device, 'awgs/0/dio/valid/polarity', 2) # commented out because already in other function
    #seti(daq, device, 'awgs/0/dio/valid/index', 16)

    #daq.setInt(f'{device}/dios/0/mode', 1) #2
    #daq.setInt(f'{device}/raw/dios/0/testmode', 2)
    #daq.setDouble(f'{device}/dios/0/drive', 0x3)
    #daq.setDouble(f'{device}/dios/0/extclk', 0x2)
    #daq.setInt(f'{device}/sigouts/*/on', 1)


    ##################################################
    # b) UHF AWG taking trigger and running the sequence
    ##################################################

    uhf_awg_source_string = """
        while(1){
        waitDIOTrigger();
        // waitAnaTrigger(1, 0x7FF);
        // playWave(fake_qubit_waveform);
        // setTrigger(1);

        // Set address to 0 (which writes to registers 0-9)
        setID(0);
        // Set mask to 0xF, which means only readout bits 0-3 will be read and only registers 0-3 will be updated
        startQAResult(0xF << 16);

        // startQAResult();
        // waitQAResultTrigger();

        // wait for feedback to propagate through system
        // wait(1400);

        // Trigger scope
        setTrigger(1);
        setTrigger(0);
        //waitDIOTrigger(); // Don't set it. Messes up the feedback - why?
        }
        """

    ####################################
    # Upload waveform onto UHF
    ###################################
    print ('starting to upload triggering waveform onto UHF AWG')

    h = daq.awgModule()
    h.set('awgModule/device', UHFQA)
    #h.set('awgModule/index', 1)
    h.execute()

    h.set('awgModule/compiler/sourcestring', uhf_awg_source_string)

    # Start Compiler
    while h.getInt('awgModule/compiler/status') == -1:
        time.sleep(0.1)
        print('.', end='')
    if h.getInt('awgModule/compiler/status') == 0:
        print("Compiler successful!")
    else:
        raise Exception("Compiler ERROR: "+str(h.getInt('awgModule/compiler/status')) + " " +str(h.getString('awgModule/compiler/statusstring')))

    # Upload ELF
    h.set('awgModule/elf/upload', 1)
    while h.getInt('awgModule/elf/upload') == 1:
        time.sleep(0.1)
        print('.', end='')
    elfstatus = h.getInt('awgModule/elf/status')
    if elfstatus == 0:
        print("ELF upload successful!")
    else:
        raise Exception("ELF upload ERROR! " + str(elfstatus))

    daq.setInt(f'{device}/awgs/0/single', 1)
    daq.setInt(f'{device}/awgs/0/enable', 1)

    ##############################################################
    # c) setting scope trigger on waiting for AWG trigger
    #############################################################

    trigreference = s['trigreference']
    scope_trigholdoff = s['scope_trigholdoff']

    # here, there are more settings compared to
    # the backpanel setting in the triggering setup
    # especially the qas and dios settings
    parameters = [
        # Triggering
        ('scopes/0/trigchannel', 192), # set the make scope recieving AWG 1 Trigger
        ('scopes/0/trigslope', 1),
        ('scopes/0/trigenable', 1),
        ]

    dparameters = [
        ('scopes/0/trigholdoff', scope_trigholdoff),
        ('scopes/0/trigreference', trigreference),
        ('scopes/0/triglevel', 200),
        ]

    for node, key in parameters:
        daq.setInt(f'{device}/{node}', key)

    for node, key in dparameters:
        daq.setDouble(f'{device}/{node}', key)

def generate_QA_results(daq, device, UHFQA_settings):
    """
    Setting the QA-result to an emulated measurement of an excited qubit state.
    """
    parameters = [
        # Trigger the QA result with the Trigger coming from AWG
        ('qas/0/integration/trigger/channel', 7), # or 0 if directly
        # Bypass crosstalk to reduce latency
        ('qas/0/crosstalk/bypass', 1),
        # Reset QA results (useful if there were previous errors)
        ('qas/0/result/reset', 1)
    ]

    dparameters = [
        ('qas/0/integration/length', 16),
    ]

    for node, key in parameters:
        daq.setInt(f'{device}/{node}', key)

    for node, key in dparameters:
        daq.setDouble(f'{device}/{node}', key)

    for i in range(5):
        daq.setDouble(f'/{device}/qas/0/thresholds/{i}/level', -100)
        daq.setDouble(f'/{device}/qas/0/thresholds/{i+5}/level', 100)

def set_DIO_to_feedback(daq, UHFQA, UHFQA_settings):
    """
    Setting the QA-result to an emulated measurement of an excited qubit state.
    """
    device = UHFQA
    dev_uhf = UHFQA

    parameters = [
        # Set DIO output to QA result
        ('dios/0/mode', 2),
        # waitDIOTrigger should _not_ work in obsolete strobe mode
        ('awgs/0/dio/strobe/slope', 0),
        # "high" state of DIO bit 16 indicates valid trigger event for waitDIOTrigger
        ('awgs/0/dio/valid/polarity', 2),
        ('awgs/0/dio/valid/index', 16),
        # Switch to QCCS mode
        ('raw/dios/0/testmode', 2), #2
        ]

    dparameters = [
        #('qas/0/integration/length', 128),
        ('triggers/in/0/level', 1),
        #('dios/0/drive', 0x3),
        # Drive the two least significant bytes
        ('dios/0/drive',  0b0011),
        ('dios/0/extclk', 0x2)
        ]

    for node, key in parameters:
        daq.setInt(f'{device}/{node}', key)

    for node, key in dparameters:
        daq.setDouble(f'{device}/{node}', key)


def DIO_AWG_Trig(daq, UHFQA, AWG_settings):
    """
    The DIO trigger triggers the AWG to emmit the waveforms
    """
    device = UHFQA
    s = AWG_settings
    gap_wait = s['gap_wait']
    sequence_wait = gap_wait*4
    awg_length = s['awg_waveform_length']
    ###################################################
    # a) DIO trigger in triggering UHF AWG
    ###################################################
    #seti(daq, device, 'awgs/0/dio/strobe/slope', 0) # Todo: maybe put to function forward_QA_results_with_DIO
    #seti(daq, device, 'awgs/0/dio/valid/polarity', 2) # commented out because already in other function
    #seti(daq, device, 'awgs/0/dio/valid/index', 16)

    #daq.setInt(f'{device}/dios/0/mode', 2) # 2
    #daq.setInt(f'{device}/raw/dios/0/testmode', 2)
    daq.setInt(f'{device}/triggers/in/0/level', 1)
    #daq.setDouble(f'{device}/dios/0/drive', 0x3) 
    #daq.setDouble(f'{device}/dios/0/drive', 2)
    # Drive the two least significant bytes
    #daq.setDouble(f'{device}/dios/0/drive',  0b0011),
    #daq.setDouble(f'{device}/dios/0/extclk', 0x2)
    daq.setInt(f'{device}/sigouts/*/on', 1)
    #('dios/0/drive', 2),
    #('dios/0/drive', 0x3),

    daq.setInt(f'{device}/awgs/0/dio/strobe/slope', 0)
    daq.setInt(f'{device}/awgs/0/dio/valid/polarity', 2)
    daq.setInt(f'{device}/awgs/0/dio/valid/index', 16) #?
    ##################################################
    # b) UHF AWG taking trigger and running the sequence
    ##################################################
    waveform = 'drag'
    uhf_awg_source_string = """
            const AWG_N = _c1_;
            const seq_wait = _wait_;
            const gap_wait = _gapwait_;
            //const waveform = drag;
            wave w = _waveform_(AWG_N, AWG_N/2, AWG_N/8); // gaussian pulse with (length, center, width)
            wave w0 = zeros((seq_wait));
            wave w1 = zeros((gap_wait));
            while(1){
                waitDIOTrigger(); // the trigger coming from the ZSync input, together with the settings in initialize_awg
                playWave(w0);
                playWave(1,w);
                playWave(w1);
                playWave(2,w);
                }
        """
    uhf_awg_source_string = uhf_awg_source_string.replace('_c1_', str(awg_length))
    uhf_awg_source_string = uhf_awg_source_string.replace('_wait_', str(32))
    uhf_awg_source_string = uhf_awg_source_string.replace('_gapwait_', str(gap_wait))
    uhf_awg_source_string = uhf_awg_source_string.replace('_waveform_', str(waveform))

    ####################################
    # Upload waveform onto UHF
    ###################################
    print ('starting to upload triggering waveform onto UHF AWG')

    h = daq.awgModule()
    h.set('awgModule/device', UHFQA)
    #h.set('awgModule/index', 1)
    h.execute()

    h.set('awgModule/compiler/sourcestring', uhf_awg_source_string)

    # Start Compiler
    while h.getInt('awgModule/compiler/status') == -1:
        time.sleep(0.1)
        print('.', end='')
    if h.getInt('awgModule/compiler/status') == 0:
        print("Compiler successful!")
    else:
        raise Exception("Compiler ERROR: "+str(h.getInt('awgModule/compiler/status')) + " " +str(h.getString('awgModule/compiler/statusstring')))

    # Upload ELF
    h.set('awgModule/elf/upload', 1)
    while h.getInt('awgModule/elf/upload') == 1:
        time.sleep(0.1)
        print('.', end='')
    elfstatus = h.getInt('awgModule/elf/status')
    if elfstatus == 0:
        print("ELF upload successful!")
    else:
        raise Exception("ELF upload ERROR! " + str(elfstatus))

    daq.setInt(f'{device}/awgs/0/single', 1)
    daq.setInt(f'{device}/awgs/0/enable', 1)

    sleep(7)

    #######################################
    # c) trigger settings
    #######################################
    #daq.setInt(f'{device}/awgs/0/triggers/0/level', 0.2)
