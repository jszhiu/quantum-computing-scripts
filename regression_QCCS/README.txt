ReadMe QCCSTest

1 OVERVIEW

The QCCS Test aims at measuring the latency, jitter and skew of (several) HDAWG(s) channel outputs ZSync-Triggered by a PQSC. Their channel outputs are measured and compared on a UHF(QA).
The code can be extended to measure these three (latency, jitter and skew) for another set of multi-instrument setup triggered in some way.




2 SETUP

The ZSync-connection between PQSC and the HDAWGs under test need to be connected. Furthermore, the HDAWG's channels that should be tested need to be combined into maximally two cables, which can be
	connected to the UHF input channels.
For the time being, the PQSC Trigger output is connected to the UHF Trigger input 1. At a later stage, the triggering of the UHF may be done differently. 
Lastly, the PQSC needs to be connected to an external clock.




3 RUNNER FILES OVERVIEW

There are three Runner files.
The Run_Plural.py file is the main runner file. It calls the multiple_run function.
The Run_getting_and_preplotting_raw_data.py is intended to give an overview on the measured signal on the scope 
in a brief way, without having to go through the long code. It is therefore basically meant to help debugging. It furthermore separates the measured data if entered in the first or second UHF input.
The RepeatedRun.py file contains functions that put a further layer ontop of the Run_Plural.py. It repeats the multiple_waves function and all that comes with it for a certain amount of times,
within run_Time_Evolution. After that, RepeatedRun_analysis generates an analysis on the the results of all these runs, meaning the evolution in time.

In any of these cases, an arbitrary number of channels can be chosen out of an arbitrary number of HDAWGs, and the waveform is a 'drag' function
(the first derivative of the gaussian function). Furthermore, the number of runs upon which an average skew and a jitter should be calculated of can be set. In addition to that the bool value
'regression' indicates if the setup is done with the cabeling as in the in-house regression setup. The warmup_time parameter indicates the warmup time in minutes.
 The init_experiment string variable inidcates, which initial setting is chosen to measure the three times latency, jitter and skew.

In any of these cases, the experiment needs to be initiated with init_zsync_trigger_experiment or init_zsync_feedback_experiment in the init_and_disconnect_skew_msrmnt.py file. Each of these
intialisation functions set the PQSC, the HDAWGs and the UHF to the desired settings by calling smallest entity functions in the setting_functions.py file.
Every such experiment setup has:
0) if needed: the initialization of the clocking UHF, sending the clocks to all the other connected devices. If there is none, the clocking needs to be done in some different way.
1) the initialization of the PQSC and output of triggers to the connected HDAWGs and the connected UHFs.
2) the initialization of the HDAWGs including the upload of the waveforms onto the AWG FPGAs and receiving a trigger.
3) the initialization of the measuring UHF. It is the one in which the channel inputs are connected to the HDAWGs outputs.

After initialization, the measuring UHF's scope data are either shown (in the case of Run_getting_and_preplotting_raw_data.py) or proceeded as follows:
Within multiple_waves function, first functions within the UHFQA_Run_Plural_Gathering.py file are called, which gather UHF scope data and analyse on-the-run. The analysis consists of fitting a
drag function onto each measured drag waveform, deducing the center of this waveform. Then, after the data has been gathered, the multiple_waves function calles UFQA_Plural_Analysis.py functions,
which do the analysis on the gathered data, meaning jitter and skew calculations, by comparing the absolute deviation of the assumed difference in time between two waveform centers to the measured
difference in time.
It has not been written ideally, since if no data is recieved, resulting in a 'None', the code stops. However, at a rerun, it will work again.





4 OUTPUT

The output depends on what which runner file is called. 1) Run_getting_and_preplotting_raw_data.py does not give any saved output, 2) Run_Plural.py outputs a folder with the "date&time&multiple" and
 3) RepeatedRun.py outputs a "RepeatedRun&date&time&" folder with "date&time&multiple" folders containing the run of one Run_Plural.py output.

4.1 Run_Plural.py

There are two output packets: the first during the data gathering in UHFQA_Plural_Gathering and the second in the data analysis in UHFQA_Plural_Analysis.

In the first package, the plotshow_settings can be set to outputting the following:
The plot of what the UHF scope measured can be saved in a "peaksplot" plot for every run. It shows the whole scope-data of each run.
The wavefit of how the waveforms's position was found can be saved for both of the two channels in the "gauss_fit" resp. "drag_fit" plots.
If needed, a plot of the gaussians (but no saving, only showing) can be set.
The raw_data saves the results of the 'gauss_fit' or 'drag_fit' in a json file. It contains the found waveform peak (gauss) / waveform zero-crossing (drag) in the 'ts_all' matrix with the error
corresponding to each peak in the 'ts_all_error' matrix. They are both of shape (n_average, n_channels) with n_average the total amount of runs to find the jitter and n_channels the total amount of
channels under observation. Other thins saved in the raw_data json file include the UHF, PQSC, HDAWGs device names, the channels_list showing the list of lists corresponding to the chosen
HDAWG output channels and n_average, which is the amount of runs that are taken to find the jitter.

In the second package, the calculated analysis of the peaks found in the gathering are saved. Note that there are two ways of channel numbering: one counting up within a HDAWG device and then starting
at zero again, and one counting through all the channels in the order of the initialized HDAWGS in the HDAWGS folder. 
The peak position of every channel as it is measured by the UHF is saved in "channel_msrmts". The HD device name and its channel are given in the figure filename and the figure title.
The jitter within each channel, calcualted in two ways, is saved in the "Jitter" figure plot. The channel number is increasing over all HDAWGs, so without restarting at zero at a new HDAWG. It could be
possible, that when scaling up to many HD's, the saving name is too long.
The time locations of all the measured peaks can be saved into the "Peaks_Time_Location".
The skew for every combination of channels is saved into the matrix shown in "Skew_Matrix". For two ways of calculating the jitter - by std. deviation and by max. difference between two measured points -
the fluctuation in skew is saved into the matrices the two figures "Skew_fluct_Std" and "SkewFluct_Max_Diff".
The "analysis_data" json file contains i) the peak_locations (in time), ii) the jitter within each channel (including errors), iii) the skew matrix and iv) the two skew_fluctuations = cross channel
jitter matrices.

4.2 RepeatedRun.py

The RepeatedRun.py calls the 4.2 Multiple run as often as set in the initial settings by running the run_Time_Evolution function, therefore generating that amount of subfolders with the content from 4.2.
The RepeatedRun_analysis function then runs through all the subfolders and processes the data saved in "analysis_data" by saving the time evolution of jitter and skew in figures and in the
"RepeatedRun_data_results".










