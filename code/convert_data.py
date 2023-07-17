# -*- coding: utf-8 -*-
import glob, os, json, shutil
import numpy as np
import pandas as pd
import struct
from scipy.signal import butter, filtfilt

# plots---------------------------
import matplotlib.pyplot as plt
import seaborn as sns
# extract fct: 
# plotting
#

class Convert_data:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    '''
    
    def __init__(self, config):
        self.config = config # load config info
        self.subject_names= config["list_subjects"]
        self.rawdata_dir=config["main_dir"] +config["rawdata_dir"]
        self.data_dir=config["main_dir"] + config["data_dir"]
        self.session_names=self.config["session_names"]
        self.runs=self.config["subjects_acq"]
        
        
        #>>> create individual output directory if needed -------------------------------------
        for subject_name in self.subject_names:
            if not os.path.exists(self.data_dir + "/" + subject_name):
                    os.mkdir(self.data_dir + "/" + subject_name) # main subject folder
                    for sess in self.session_names:
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/" ) # MSL or MA folder
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/calib/" ) # calibration folder
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/fam1/" ) # familiarization 1 folder
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/fam2/" ) # familiarisation 2 folder
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/RNDpre/" ) # random before the test folder
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/"+sess+"/" ) # test folder
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/RNDpost/" ) # random after the test folder
            else:
                
                print("Data for subjects " + subject_name + " can be found here: " + self.data_dir + "/" + subject_name)
        
        #>>> Copy and rename rawdata -------------------------------------
        for subject_name in self.subject_names:

            for sess_nb in range(0,len(self.config["list_subjects"][subject_name])):
                sess=self.config["list_subjects"][subject_name]["sess0" + str(sess_nb+1)] # name of the session for that participant

                # Calibration files --------------------------------------
                calib_card_files_raw=sorted(glob.glob(self.rawdata_dir  + "/"+ subject_name + "/sess0" + str(sess_nb+1) +"/calib*_cardinals.dat"))
                calib_mvmnt_files_raw=sorted(glob.glob(self.rawdata_dir  + "/"+ subject_name + "/sess0" + str(sess_nb+1) +"/calib*_movements.dat"))
                calib_txt_files_raw=sorted(glob.glob(self.rawdata_dir  + "/"+ subject_name + "/sess0" + str(sess_nb+1) +"/calib*.txt"))
                for file_nb in range(len(calib_card_files_raw)):
                    if len(calib_card_files_raw) >1:
                        tag="_run-0" + str(file_nb+1)
                    else:
                        tag=""
                    calib_card_files=self.data_dir + "/" + subject_name + "/sess_"+ sess + "/calib/" + subject_name + "_calib"+ tag +"_cardinals.dat"
                    calib_mvmnt_files=self.data_dir + "/" + subject_name + "/sess_"+ sess + "/calib/" + subject_name + "_calib"+ tag + "_movements.dat"
                    calib_txt_files=self.data_dir + "/" + subject_name + "/sess_"+ sess + "/calib/" + subject_name + "_calib" + tag  + ".txt"
                   
                    if not os.path.exists(calib_card_files):
                        shutil.copy(calib_card_files_raw[file_nb],calib_card_files)
                        shutil.copy(calib_mvmnt_files_raw[file_nb],calib_mvmnt_files)
                        shutil.copy(calib_txt_files_raw[file_nb],calib_txt_files)

                # Acqusition files --------------------------------------
                acq_mvmnt_files_raw=sorted(glob.glob(self.rawdata_dir  + "/"+ subject_name + "/sess0" + str(sess_nb+1) +"/sub*_movement.bin"))
                acq_schedule_files_raw=sorted(glob.glob(self.rawdata_dir  + "/"+ subject_name + "/sess0" + str(sess_nb+1) +"/sub*_schedule.dat"))
                acq_trials_files_raw=sorted(glob.glob(self.rawdata_dir  + "/"+ subject_name + "/sess0" + str(sess_nb+1) +"/sub*_trial.dat"))
                
                for file_nb in range(len(acq_mvmnt_files_raw)):
                    run_name=self.runs[subject_name]["sess0" + str(sess_nb+1)][file_nb]
                    acq_mvmnt_files=self.data_dir + "/" + subject_name + "/sess_"+ sess + "/" + run_name + "/" + subject_name + "_sess0"+str(sess_nb+1) + "_" + run_name +"_movement.bin"
                    acq_schedule_files=self.data_dir + "/" + subject_name + "/sess_"+ sess + "/" + run_name + "/" + subject_name + "_sess0"+str(sess_nb+1) + "_" + run_name +"_schedule.dat"
                    acq_trial_files=self.data_dir + "/" + subject_name + "/sess_"+ sess + "/" + run_name + "/" + subject_name+ "_sess0"+str(sess_nb+1) + "_" + run_name +"_trial.dat"
                    
                    if not os.path.exists(acq_mvmnt_files):
                        shutil.copy(acq_mvmnt_files_raw[file_nb],acq_mvmnt_files)
                        shutil.copy(acq_schedule_files_raw[file_nb],acq_schedule_files)
                        shutil.copy(acq_trials_files_raw[file_nb],acq_trial_files)
    
    
                    
    def read_movement(self,subject_name, data_filename=False):
        '''
        The read_movement fucntion is used to read binary files that incorporate joystick movement recordings
        Inputs
        ----------
        subject_name: string
            name of the participant
        data_filename: string
            data to read (.bin)
    
        Returns
        ----------
        data_extracted: array
            recordings data in array format
        df: dataframe
            recordings data in array format
        '''
        
        data2plot={};
        column_names = ["axis_states_x", "axis_states_y", "axis_raw_x", "axis_raw_y", "t_since_start", "device_time", "sample", "trial", "phase"]
        # "axis_states_" : is the position after processing this packet (and applying calibration)
        # axis_raw_x value : is the raw x,y reading prior to applying calibration
        # sample: enotes the sample number, which actually counts axis packets. So between two subsequent MOVEMENT entries there may be multiple samples.
        # device_time # The latest known device time
        data_array = [] # create empty array
        data_extracted=np.empty((1,9)) # create empty array with 9 columns
                
        # read movement data:
        with open(data_filename, 'rb') as file:
            while True:
                binary_data = file.read(36)  # Read 36 bytes for each line
                if not binary_data:
                    break  # Exit the loop if there are no more lines to read

                data = struct.unpack('ffffffiii', binary_data)
                data_array=np.array(data).T

                data_extracted=np.concatenate((data_extracted,data_array[np.newaxis,:]),axis=0) # Unpack the data for each line

                # save data in a dataframe format
                df= pd.DataFrame(data_extracted,columns=column_names)
                
        return data_extracted, df

    def filter_movement(self,dataframe,cutoff_frequency=10,fs=400):
       
        x_signal=dataframe["axis_raw_x"]
        y_signal=dataframe["axis_raw_y"]
        # Design the Butterworth filter
        order = 4  # The order of the filter
        b, a = butter(order, cutoff_frequency, btype='low', analog=False,fs=fs)

        # Apply the filter to the signal
        filtered_signal_x = filtfilt(b, a, x_signal)
        filtered_signal_y = filtfilt(b, a, y_signal)
        
            

    def plot_movement(self, subject_name, data_df=None,sample_rate=0.0025,save_plot=False,plot_filename=None):
        '''
        The plot_movement fucntion is used to plot x and y axis of the movement recordings
        Inputs
        ----------
        subject_name: string
            name of the participant
        data_df: dataframe
            data to read (.bin)
        sample_rate: int
            data sample rate (default=0.01)
        
        save_plot: bool
            to save the plot as a png file (default=False)
            
        plot_filename: string
            filename of the plot (default=None)
    
        '''
            
        # Initiate
        plt.figure(figsize=(15,6))
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks",rc=custom_params)

        sns.lineplot(x=data_df["sample"]*sample_rate,y="axis_raw_x",color='#2a9675', data=data_df,linewidth=2.5)
        sns.lineplot(x=data_df["sample"]*sample_rate,y="axis_raw_y",color='#e3a32c', data=data_df,linewidth=2.5)

        plt.ylabel('joystick position')
        plt.xlabel('time (in sec)')

        
        custom_legend = [plt.Line2D([], [], color='#2a9675', label='x axis'),
            plt.Line2D([], [], color='#e3a32c', label='y axis')]

        plt.legend(handles=custom_legend)
        
        if save_plot==True:
            if plot_filename==None:
                plot_filename=self.data_dir + "/" + subject_name + "_movement.png"
                
            else:
                plot_filename=plot_filename

            plt.savefig(plot_filename, dpi=300) 


                       
       