## -----------------------------------------------------------------------------
## @brief Helper Functions for Initialisations
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Defining  HDAWG specific utility functionscalled by init_*.py:
### a) make_sourcestrings
### b) initialize_awg(daq, device_names, awgs_list, channels_list, awg_source_string)
### c) activate_external_reference_hdawg(devices, source)
### d) deactivate_external_reference_hdawg(devices)
### e) set_HDAWG_channels(daq, device, channels_list[i])
## ------------------------------------------------------------------------------
import logging
import time

logger = logging.getLogger()

def make_sourcestring_trigger(awg_length, gap_wait, waveform, number_of_HDAWGs):
    '''
    Generates the list of sourcestrings in SeqC which can later be uploaded onto the AWGS of the HDAWGs.

    Input:  awg_length      : waveform length (smallest possible: 32 points) (int)
            gap_wait        : length of waveform containing zeros between channels connected to same AWG (int), ideally the same as awg_length
            sequence_wait   : length of waveform containing zeros between two following AWGs (int), ideally = gap_wait*4
            waveform        : shape of waveform, either 'gauss' or 'drag'
            number_of_HDAWGs : number of HDAWGS under test (int)

    Output: awg_source_string_list


    Generates the list of list of sourcestrings which can later be uploaded onto the AWGS of the HDAWGs. The list will have number_of_HDAWGs entries, one for each device. Each of these number_of_HDAWGs list entries then furthermore is a list of 4 entries containing the strings for each AWG within that HDAWG in 4x2 channelgrouping.
    '''

    number_of_AWG_per_HD = 4 # dependent on channelgrouping, here: 4x2
    sequence_wait = gap_wait*4 # such that differences between the AWGs are correct in 4x2 channelgrouping

    for i in range(number_of_HDAWGs): # looping over HDAWGs
        for j in range(number_of_AWG_per_HD): # looping over number of channels per HDAWG
            vars()['awg_source_string_{}_{}'.format(i,j)] = \
            """
            const AWG_N = _c1_;
            const seq_wait = _wait_;
            const gap_wait = _gapwait_;
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

    # Replace the placeholder with the integer constants.
    awg_source_string_list = []
    for i in range(number_of_HDAWGs): # looping over HDAWGs
        awg_source_string_list_HDi = []
        for j in range(number_of_AWG_per_HD): # looping over number of channels per HDAWG
            #pdb.set_trace()
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_c1_', str(awg_length))
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_wait_', str((32+ i*4*sequence_wait + sequence_wait*j)))
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_gapwait_', str(gap_wait))
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_waveform_', str(waveform))
            awg_source_string_list_HDi.append(vars()['awg_source_string_{}_{}'.format(i,j)])
        awg_source_string_list.append(awg_source_string_list_HDi)

    return awg_source_string_list


def make_sourcestring_feedback(awg_length, gap_wait, waveform, number_of_HDAWGs):
    '''
    Generates the list of sourcestrings in SeqC which can later be uploaded onto the AWGS of the HDAWGs.

    Input:  awg_length      : waveform length (smallest possible: 32 points) (int)
            gap_wait        : length of waveform containing zeros between channels connected to same AWG (int), ideally the same as awg_length
            sequence_wait   : length of waveform containing zeros between two following AWGs (int), ideally = gap_wait*4
            waveform        : shape of waveform, either 'gauss' or 'drag'
            number_of_HDAWGs : number of HDAWGS under test (int)

    Output: awg_source_string_list


    Generates the list of list of sourcestrings which can later be uploaded onto the AWGS of the HDAWGs. The list will have number_of_HDAWGs entries, one for each device. Each of these number_of_HDAWGs list entries then furthermore is a list of 4 entries containing the strings for each AWG within that HDAWG in 4x2 channelgrouping.
    '''

    number_of_AWG_per_HD = 4 # dependent on channelgrouping, here: 4x2
    sequence_wait = gap_wait*4 # such that differences between the AWGs are correct in 4x2 channelgrouping
    for i in range(number_of_HDAWGs): # looping over HDAWGs
        for j in range(number_of_AWG_per_HD): # looping over number of channels per HDAWG
            vars()['awg_source_string_{}_{}'.format(i,j)] = \
            """
            const AWG_N = _c1_;
            const seq_wait = _wait_;
            const gap_wait = _gapwait_;
            wave w = _waveform_(AWG_N, AWG_N/2, AWG_N/8); // gaussian pulse with (length, center, width)
            wave w0 = zeros((seq_wait));
            wave w1 = zeros((gap_wait));

            while(1){
                waitDIOTrigger(); // the trigger coming from the ZSync input, together with the settings in initialize_awg
                wait(2);
                waitDIOTrigger();
                // getDIOTriggered();
                playWave(w0);
                playWave(1,w);
                playWave(w1);
                playWave(2,w);
                // setUserReg(0, (getDIOTriggered() >> 1) & 0x3ff); // setting the mask to the qubit results
                }

            """

    # Replace the placeholder with the integer constants.
    awg_source_string_list = []
    for i in range(number_of_HDAWGs): # looping over HDAWGs
        awg_source_string_list_HDi = []
        for j in range(number_of_AWG_per_HD): # looping over number of channels per HDAWG
            #pdb.set_trace()
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_c1_', str(awg_length))
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_wait_', str((32+ i*4*sequence_wait + sequence_wait*j)))
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_gapwait_', str(gap_wait))
            vars()['awg_source_string_{}_{}'.format(i,j)] = vars()['awg_source_string_{}_{}'.format(i,j)].replace('_waveform_', str(waveform))
            awg_source_string_list_HDi.append(vars()['awg_source_string_{}_{}'.format(i,j)])
        awg_source_string_list.append(awg_source_string_list_HDi)

    return awg_source_string_list

def set_dio_mode_zsync(daq, device):
    '''
    Setting the (HDAWG) trigger input to ZSync input.
    '''
    daq.setInt(f'/{device}/DIOS/0/MODE', 3)

def initialize_awg(daq, device, awg_source_string_list, HDAWG_settings):
    '''
    Initializing a HDAWG device settings, upload AWG sourcestring and setting input trigger to ZSync (by calling set_dio_mode_zsync). Not yet enabling the output signals.

    Input:  daq:            current run in data aquisition
            device:         HDAWG device name
            awg_source_string_list: list of 4 SeqC sourcestring entries for each AWG in 4x2 channelgrouping.
            HDAWG_settings: dictionary containing how the HDAWG settings should be initialized.

    Output: None

    '''
    print('Initialize AWG on '+device)
    s = HDAWG_settings
    awg_rate = s['awg_sampling_rate_variable']
    awg_single_shot = s['awg_single_shot']
    channelgrouping = s['channelgrouping']

    #amplitude = 2
    #sigouts_amplitude=0.5
    #sigouts_range=1.5

    # Set AWG channel grouping:
    #seti(daq, device, 'system/awg/channelgrouping', channelgrouping)
    daq.setInt(f'{device}/system/awg/channelgrouping', channelgrouping)

    params = [
        ('sigouts/*/imp50', 1),
        # ('awgs/0/dio/strobe/slope', 0),
        #('awgs/0/dio/valid/polarity', 2),

        # Set all marker output to low.
        ('triggers/out/*/source', 18),
        ('sines/*/enables/1', 0)

    ]

    for node, key in params:
        #seti(daq, device, node, key)
        daq.setInt(f'{device}/{node}', key)

    #########################################
    ## Initialize AWG(s)
    #########################################
    # AWG in direct mode
    # setd(daq, device, 'awgs/'+str(awg)+'/outputs/*/mode', 0)
    # DIO Trigger Settings

    awgs_list = []
    dev_type = daq.getString(f'/{device}/features/devtype')
    if dev_type == 'HDAWG8':
        awgs_list = [0,1,2,3]

    if dev_type == 'HDAWG4':
        awgs_list = [0,1]
        awg_source_string_list = awg_source_string_list[:2]

    for awg in awgs_list:
        #seti(daq, device, 'awgs/'+str(awg)+'/dio/strobe/slope', 0)
        daq.setInt(f'{device}/awgs/{awg}/dio/strobe/slope', 0)
        # setd(daq, device, 'awgs/'+str(awg)+'/dio/valid/index', 1)
        daq.setDouble(f'/{device}/awgs/{awg}/dio/valid/index', 0)
        daq.setInt(f'/{device}/awgs/{awg}/dio/valid/polarity', 2)
        #seti(daq, device, 'awgs/'+str(awg)+'/dio/valid/polarity', 2)
        #setd(daq, device, 'awgs/'+str(awg)+'/dio/mask/value', 0x3ff) ##here
        #setd(daq, device, 'awgs/'+str(awg)+'/dio/mask/shift', 0)
        # Set sampling rate to 2.4 GHz:
        #seti(daq, device, 'awgs/'+str(awg)+'/time', awg_rate)
        daq.setInt(f'/{device}/awgs/{awg}/time', awg_rate)
        # set AWG in single-shot mode.
        #seti(daq, device, 'awgs/'+str(awg)+'/single', awg_single_shot)
        daq.setInt(f'/{device}/awgs/{awg}/single', awg_single_shot)
        #seti(daq, device, 'awgs/'+str(awg)+'/enable', 0)
        daq.setInt(f'/{device}/awgs/{awg}/enable', 0)

    #########################################
    ## Upload Waveform to all used AWGs
    #########################################
    #awg_source_string_list = [awg_source_string, awg_source_string, awg_source_string, awg_source_string]

    for awg, awg_source_string in zip(awgs_list, awg_source_string_list):
        print("STARTING AWG" + str(awg))
        h = daq.awgModule()
        h.set('awgModule/device', device)
        h.set('awgModule/index', awg)
        h.execute()

        h.set('awgModule/compiler/sourcestring', awg_source_string)

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

        daq.setInt(f'/{device}/awgs/{awg}/enable', 1)

# Lock HDAWG devices to external reference clock
def set_external_reference_hdawgs(daq, devices, source):
    '''
    Setting the HDAWGs clock and the DIO clock to the ZSYNC reference by setting source = 2.
    Deactivating the external clock by setting source = 0.

    Setting external clock of DIO clock to 0x1
    '''
    print("Set reference-clock to external")
    for device in devices:
        #seti(daq, device, 'system/clocks/referenceclock/source', source)
        daq.setInt(f'{device}/system/clocks/referenceclock/source', source)
        # seti(daq, device, 'raw/dios/0/extclk', 0x1)
    # wait until both status are 0 and the sources correct,
    # or timeout occurs:
    poll_cnt = 0
    while True:
        stat = 0
        ok = True
        for device in devices:
            stat = daq.getInt(f'/{device}/system/clocks/referenceclock/status')
            src  = daq.getInt(f'/{device}/system/clocks/referenceclock/source')

            # stat1 = geti(daq, device, 'raw/dios/0/extclk/status')
            # src1  = geti(daq, device, 'raw/dios/0/extclk/source')
            if stat != 0 or src != source:
                ok = False

        # all stat need to be 0, all src need to be source
        if ok:
            print ("Done")
            return True
        poll_cnt +=1
        if poll_cnt > 10:
            print("Timeout, failed!")
            return False
        time.sleep(0.5)
        print(str(poll_cnt), end=' ')

def set_HDAWG_channels(daq, device, channels_list, regression):
    '''
    Setting and enabling the HDAWG (device) channels' output signals as given in channels_list of form [0], [0,1,2,3,4,5,6,7] etc.

    Input:  daq             : current run in data aquisition
            device          : HDAWG device name
            channels_list   : list of channels that should be outputting signals
    '''
    amplitude_range = 2 # default amplitude range
    if regression:
        dev_type = daq.getString(f'/{device}/features/devtype')
        if dev_type == 'HDAWG8':
            amplitude_range = 2

        if dev_type == 'HDAWG4' and regression == 1:
            amplitude_range = 4

        else:
            amplitude_range = 2


    # Disable all outputs from a potential previous run
    #setd(daq, device, 'sigouts/*/on', 0)
    daq.setDouble(f'/{device}/sigouts/*/on', 0)

    # Turn on outputs
    for channel in channels_list:
        #setd(daq, device, 'sigouts/'+str(channel)+'/on', 1)
        daq.setDouble(f'/{device}/sigouts/{channel}/on', 1)

        # enable direct output for wave output. FROM CAN
        #seti(daq, device, 'sigouts/'+str(channel)+'/direct', 0)
        #seti(daq, device, 'sigouts/'+str(channel)+'/range', amplitude_range) # amplitude range of output in V
        daq.setInt(f'/{device}/sigouts/{channel}/range', amplitude_range)

        #seti(daq, device, 'sigouts/'+str(channel)+'/on', 1)
        daq.setInt(f'/{device}/sigouts/{channel}/on', 1)

def set_ZSync_and_DIO_feedback(daq, HDAWGS):
    """
    Setting the ZSync connection and the DIO such that the two are passing info across the HD.
    """

    dev_hd = HDAWGS[0]
    daq.setDouble(f'/{HDAWGS[0]}/dios/0/drive', 12)

    ### NEW
    # Configure DIO switch
    daq.setInt(f'/{dev_hd}/raw/dios/0/mode', 1)

    # Configure DIO triggering
    daq.setInt(f'{dev_hd}/awgs/0/dio/strobe/slope', 0)
    daq.setInt(f'{dev_hd}/awgs/0/dio/valid/index', 0)
    daq.setInt(f'{dev_hd}/awgs/0/dio/valid/polarity', 2)
    daq.setInt(f'{dev_hd}/awgs/0/dio/mask/value', 0x3FF)
    #daq.setInt(f'{dev_hd}/awgs/0/dio/mask/shift', 0)

    # Setup AWG module
    awg_hd = daq.awgModule()
    awg_hd.set('device', dev_hd)
    awg_hd.set('index', 0)
    awg_hd.execute()
