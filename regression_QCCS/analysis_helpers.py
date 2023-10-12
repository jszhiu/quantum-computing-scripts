## -----------------------------------------------------------------------------
## @brief Helper functions used after the actual run
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Helper functions for the analysis
## ------------------------------------------------------------------------------
import logging
import time

logger = logging.getLogger()

def helper_hdawg_get_zsync_dlycal(daq, devs):
    ZSyncInfoPrint = ''
    for dev_hd in devs:
        nodes = ['dly']
        for node in nodes:
            daq.getAsEvent(f'/{dev_hd}/raw/zsync/{node}')
        data = daq.poll(0.5, 100, 0, True)

        out = ''
        for node in nodes:
            node_string = f'/{dev_hd}/raw/zsync/{node}'
            dt = data[node_string]
            node_value = data[node_string]['value']
            #node_value = data[node_string][0]
            #print(f'{node} \t= 0x{node_value:x}')
            out = out + f'   {node_string}: {node_value}'
        #print('delaystatus = phase_align_done & phase_align_valid & locked_i')
        ZSyncInfo = f'ZSync Delay calibration device {dev_hd}:\n{out}'
        logger.info(ZSyncInfo)
        ZSyncInfoPrint = ZSyncInfoPrint + '\n' + ZSyncInfo

    return ZSyncInfoPrint

def wait_busy(daq, path, dt=0.1, timeout=10., perform_sync=True):
    """ Wait for reset of busy flag. """
    wait_time = 0.0
    while True:
        if perform_sync:
            daq.sync()
        else:
            time.sleep(0.1)
        busy = daq.getInt(path)
        if not busy:
            break
        if wait_time > timeout:
            raise Exception('Timed out while waiting for busy flag {}'.format(path))
        time.sleep(dt)
        wait_time += dt
    return wait_time
