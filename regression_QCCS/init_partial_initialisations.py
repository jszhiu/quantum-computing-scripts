
## -----------------------------------------------------------------------------
## @brief Initialisation functions for parts of the triggering sync (and maybe
## eventually also feedback) test
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details Partial initialisations of chosen devices only, mainly called by a
## variant of a RepeatedRun.py test
## ------------------------------------------------------------------------------
import logging

import setting_HDAWGs
import setting_PQSC
import setting_UHF

logger = logging.getLogger()

def init_zsync_trigger_experiment_UHFQA(daq, UHFQA, Mod, init_settings, reinit_settings):
    #disconnect

    if reinit_settings == 'full':
        # reconnect
        INTERFACE='1gbe'
        daq.connectDevice(UHFQA, INTERFACE)
        logger.info(f'Done connecting to measuring UHF {UHFQA}')
        #activate_external_reference_uhf(daq, UHFQA, 0)
        daq.setInt(f'{UHFQA}/system/extclk', 0)
        #daq.sync()
        logger.info(f'{UHFQA} clock set to internal')
        setting_UHF.activate_external_reference_uhf(daq, UHFQA, 1)
        #daq.sync()

    if reinit_settings == 'only_ext_clock_change':
        setting_UHF.activate_external_reference_uhf(daq, UHFQA, 1)
        #daq.sync()

    setting_UHF.initialize_uhfqa(daq, UHFQA, Mod, init_settings)
    setting_UHF.frontpanel_scope_trig(daq, UHFQA, Mod, init_settings)
    #init.backpanel_scope_trig(daq, UHFQA, Mod, UHFQA_settings)

    wave_nodepath = '/{}/scopes/0/wave'.format(UHFQA)
    #daq.sync()
    logger.info("initialize_uhfqa DONE")

    Mod.subscribe(wave_nodepath)

    return wave_nodepath

def init_zsync_trigger_experiment_HDAWG(daq, HDAWGS, init_settings, reinit_settings):
    awg_source_string_list = init_settings['awg_source_string_list']

    for i, HDAWG in enumerate(HDAWGS):
        setting_HDAWGs.initialize_awg(daq, HDAWG, awg_source_string_list[i], init_settings)
        # Set DIO to ZSync Triggering
        setting_HDAWGs.set_dio_mode_zsync(daq, HDAWG)
    logger.info("Initialization of HDAWGs DONE")

    if reinit_settings == 'full':
        for HDAWG in HDAWGS:
            daq.setInt(f'{HDAWG}/system/clocks/referenceclock/source', 0)
            logger.info(f'{HDAWG} clock set to internal')
            daq.sync()

        # HDAWGs lock reference clock to ZSYNC coming from PQSC
        setting_HDAWGs.set_external_reference_hdawgs(daq, HDAWGS, 2)
        daq.sync()

    if reinit_settings == 'only_ext_clock_change':
        # HDAWGs lock reference clock to ZSYNC coming from PQSC
        setting_HDAWGs.set_external_reference_hdawgs(daq, HDAWGS, 2)
        daq.sync()

def init_zsync_trigger_experiment_PQSC(daq, PQSC, init_settings, reinit_settings):
    daq.setInt(f'{PQSC}/execution/enable', 0)

    if reinit_settings == 'full':
        daq.setInt(f'{PQSC}/system/clocks/referenceclock/in/source', 0)
        daq.sync()
        logger.info(f'{PQSC} execution stopped and clock set to internal')
        setting_PQSC.activate_external_reference_pqsc(daq, PQSC, 1)
        daq.sync()

    if reinit_settings == 'only_ext_clock_change':
        setting_PQSC.activate_external_reference_pqsc(daq, PQSC, 1)
        daq.sync()

    setting_PQSC.init_pqsc(daq, PQSC, init_settings)
