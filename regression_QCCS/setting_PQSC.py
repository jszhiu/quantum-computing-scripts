## -----------------------------------------------------------------------------
## @brief Helper Functions for Initialisations
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Defining PQSC specific utility functionscalled by init_*.py:
### a) init_pqsc(device)
### b) activate_external_reference_pqsc (device, source)
### c) receive_and_forward(daq, PQSC)
## ------------------------------------------------------------------------------
import logging
import time

import numpy as np

logger = logging.getLogger()

############################################
## 2) PQSC specific functions
### a) init_pqsc(device)
### b) activate_external_reference_pqsc (device, source)
################################

# Initialize PQSC triggering of HDAWGs and UHF
def init_pqsc(daq, device, PQSCs):
    '''
    Initializing the PQSC with the settings given in dictionary PQSCs containing:

    Input: 'repetitions'    : number of trigger repetitions
            'timegap'       : timegap between triggering events in seconds
            'trigger_port'  : number of ZSync output to which the additional trigger output is synchronized to.
    '''
    dev_pqsc = device
    # Number of trigger repetitions
    daq.setDouble(f'/{dev_pqsc}/execution/repetitions', PQSCs['PQSC_repetitions'])
    # Holdoff time between triggers in seconds
    daq.setDouble(f'/{dev_pqsc}/execution/holdoff', PQSCs['PQSC_trigger_timegap'])

    # enable the trigger out 1
    daq.setInt(f'/{dev_pqsc}/triggers/out/0/enable', 1)
    daq.setInt(f'/{dev_pqsc}/triggers/out/0/port', 0)
    daq.setDouble(f'/{dev_pqsc}/triggers/out/0/source', 0)

    #enable the trigger out 2
    daq.setInt(f'/{dev_pqsc}/triggers/out/1/enable', 1)
    daq.setInt(f'/{dev_pqsc}/triggers/out/1/port', PQSCs['trigger_port']) # 5
    daq.setDouble(f'/{dev_pqsc}/triggers/out/1/source', 0)

    # temp: suggested from Hussein for debugging the regression HD4
    # setd(daq, device, 'RAW/QHUB/START_VALUE', 0xFF0F)
    # setd(daq, device, 'RAW/QHUB/START_VALUE', 0xFFFFFFFF)

    logger.info('Done initializing PQSC on setting functions level')

# Lock PQSC device to external reference clock
def activate_external_reference_pqsc(daq, device, source):
    '''
    Setting the PQSC clock to the external reference by setting source = 1.
    Setting the PQSC clock
    '''
    logger.info("Set reference-clock to external")
    daq.setInt(f'/{device}/system/clocks/referenceclock/in/source', source)

    # wait until both status are 0 and the sources correct,
    # or timeout occurs:
    poll_cnt = 0
    while True:
        stat = 0
        ok = True
        stat = daq.getInt(f'/{device}/system/clocks/referenceclock/in/status')
        src  = daq.getInt(f'/{device}/system/clocks/referenceclock/in/source')
        if stat != 0 or src != source:
            ok = False

        # all stat need to be 0, all src need to be source
        if ok:
            print ("Done")
            return True
        poll_cnt +=1
        """
        if poll_cnt > 10:
            print("Timeout, failed!")
            return False"""
        if poll_cnt > 100:
            assert poll_cnt > 100, "Timeout, failed!"
        time.sleep(0.5)
        print(str(poll_cnt), end=' ')

def receive_and_forward(daq, PQSC, PQSCs):
    """
     A PQSC receiving a result from any HD and forwarding it to all connected HDs via ZSync.
    """
    dev_pqsc = PQSC
    """
    # Port used
    inport = 4 # on first port --> 0 or 2
    outports = [0,1,2,3,4]
    #for outport in outports:
    # Enable (only needed if no "start execution" is used)
    # daq.setInt(f'/{dev_pqsc}/raw/zsyncs/*/enable', 1)
    # Enable feedback
    for outport in outports:
        daq.setInt(f'/{dev_pqsc}/raw/zsyncs/{outport}/feedback/enable', 1)
        # Direct feedback
        daq.setInt(f'/{dev_pqsc}/raw/zsyncs/{outport}/feedback/source', inport)
    """

    ### NEW
    # Output port used
    out_port = PQSCs['DIO_port']
    # Enable feedback on output port
    daq.setInt(f'/{dev_pqsc}/raw/zsyncs/{out_port}/txmux/fwd_en', 1)
    # Program register bank register forwarding to forward registers 0-3
    fwd_length = 4
    fwd = range(0, fwd_length)
    daq.setVector(f'/{dev_pqsc}/raw/regs/0/fwd', np.array(fwd).astype(np.uint32))
