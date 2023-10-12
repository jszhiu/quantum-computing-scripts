## -----------------------------------------------------------------------------
## @brief Putting latency measurements into histogram
## ------------------------------------------------------------------------------
## @copyright Copyright (c) 2011 Zurich Instruments AG - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and confidential
## ------------------------------------------------------------------------------
## @details called by UHFQA_Analysis if variations within one run of several
## are too large
## ------------------------------------------------------------------------------
import glob
import json
import pathlib
import statistics as stat

#import UHFQA_Plural_Analysis as pa
import matplotlib.pyplot as plt
import numpy as np

import UHFQA_Run_Plural_Gathering as UHF_pg

DIRECTORY_OF_THIS_FILE = pathlib.Path(__file__).parent

def plotdiff(path_to_file, save=False):
    title = ''
    ts_all = 0
    ts_all_error = 0

    for filename in glob.glob(str(path_to_file)+'/raw_data*.json', recursive=True):
        #print(filename)
        with open(filename) as json_file:
            files = json.load(json_file)

            ts_all = files['ts_all']
            ts_all = np.asarray(ts_all)

            ts_all_error = files['ts_all_error']
            ts_all_error = np.asarray(ts_all_error)

    for i in range(np.shape(ts_all)[1]):
        ts_all_i = ts_all[:,i]*1e12
        offset = np.floor(np.min(ts_all_i))
        ts_all_i = ts_all_i - offset

        plt.figure()
        binwidth = 40
        n, bins, patches = plt.hist(x=ts_all_i, bins=np.arange(min(ts_all_i), max(ts_all_i) + binwidth, binwidth))
        plt.title(f"Histogram of Signal Arrival for channel {i}")
        plt.xlabel("Offset-cropped latency in ps")
        run_options_in_title = f"channel_{i}"
        fig_path = path_to_file
        figname = UHF_pg.fn(str(fig_path), "Histogram_trigger_runs", run_options_in_title)
        if save:
            plt.savefig(figname, format="pdf")
        if not save:
            #plt.show()
            pass
        plt.close("all")

        overallmean = stat.mean(ts_all_i)
        mask_min = np.where(ts_all_i < overallmean)
        mask_max = np.where(ts_all_i > overallmean)
        ts_all_small = ts_all_i[mask_min]
        ts_all_large = ts_all_i[mask_max]
        diff = stat.mean(ts_all_large)-stat.mean(ts_all_small)
        diff_err = stat.mean(ts_all_error[:,i])
        nanoscale = 1e9
        print(f"Average difference {diff:.{0}f} +/- {2*diff_err*1e12:.{1}f} ps (double average error)")

def get_path(folder, supfolder=None):
    if supfolder is not None:
        path_to_file = DIRECTORY_OF_THIS_FILE / 'Output_ZSync_Trigger_Skew' / supfolder / folder
    else:
        path_to_file = DIRECTORY_OF_THIS_FILE / 'Output_ZSync_Trigger_Skew' / folder
    return path_to_file

if __name__ == '__main__':
    supfolder = 'RepeatedRun_20200608-142724-newupdate'
    folder = '20200608-143313multiple'
    folder = '20200608-143225multiple'
    folder = '20200608-145349multiple'
    folder = '20200608-145539multiple'
    folder = '20200608-145638multiple'
    path_to_file = get_path(folder, supfolder)
    plotdiff(path_to_file, False)
