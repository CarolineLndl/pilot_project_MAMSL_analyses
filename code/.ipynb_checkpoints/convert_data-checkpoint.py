# -*- coding: utf-8 -*-
import glob, os, json, shutil
import numpy as np
import pandas as pd
import struct
from scipy.signal import butter, filtfilt
import cv2
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
                        print(acq_trials_files_raw[file_nb])
                        print(run_name)
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

    def plot_trial(self,subject_name,session_name,run_name,trial_range=None,plot_indiv=False,plot_targets=True,create_movie=True):
        '''
        Return the centered and scaled data, using the calibration file
        Inputs
        ----------
        movement_file: string
            filename    
        '''
        
        # Plot movements
        ana_dir=self.config["main_dir"] + self.config["data_dir"] + "/" + subject_name + "/sess_"+ session_name + "/" + run_name 
        movement_filename=glob.glob(ana_dir + "/*_movement.csv")[0]
        trial_filename=glob.glob(ana_dir + "/*_trial.dat")[0]
        movement_data=pd.read_csv(movement_filename)

        trial_data=pd.read_csv(trial_filename)
        if trial_range==None:
            trial_range=range(0,len(trial_data))
        
        
        for trial in trial_range:
            # if ==bloc:
            plt.clf()


            # plot movemenent location
            self._plot_movement(movement_data,trial,col_tag="axis_statesfilt")
            
            if plot_targets==True:
                # plot targets location
                center = [0, 0]
                targetdirections = [0, 90, -90, 180]
                radius = .50
                targetlocation = ([(radius * np.cos(np.pi * dir / 180)) for dir in targetdirections],
                  [(radius * np.sin(np.pi * dir / 180)) for dir in targetdirections])
                plt.plot(targetlocation[0], targetlocation[1], marker="o", linestyle="None", markersize=30,alpha=0.5)
                plt.plot(center[0], center[1], marker="X", linestyle="None", markersize=10,alpha=0.5)
                
                            
            if create_movie==True:
                # save temporary plots for each trials
                tmp_dir=ana_dir + "/tmp_images/"
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                if trial<10:
                    output_file=tmp_dir + "/tmp_trial_000" + str(trial) + ".png"
                elif trial<100 and trial>9:
                    output_file=tmp_dir + "/tmp_trial_00" + str(trial) + ".png"
                elif trial<1000 and trial>99:
                    output_file=tmp_dir + "/tmp_trial_0" + str(trial) + ".png"
                    
                else:
                    output_file=tmp_dir + "/tmp_trial_" + str(trial) + ".png"
                plt.savefig(output_file, dpi=60)
        
        if create_movie==True:
            img_array=[]
            for filename in sorted(glob.glob(tmp_dir + "tmp_trial*.png")):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

                movie_file=ana_dir +"/"+ subject_name + "_"+session_name+"_"+run_name + "_trials_movie.avi"
                out = cv2.VideoWriter(movie_file,cv2.VideoWriter_fourcc(*'MJPG'), 5, size)
                #out = cv2.VideoWriter(movie_file,cv2.VideoWriter_fourcc(*'X264'), 5, size)

                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()
            shutil.rmtree(tmp_dir)
            
        
    def _plot_movement(self,movement_data,trial,col_tag):
        '''
        The plot_movement function is used to plot x and y axis of the movement
        Inputs
        ----------
        movement_data: dataframe
            dataframme with different info about the movement recordings
        trial: int
            number of the trial to plot
        col_tag: "string"
            column selected in the dataframe for plotting
        
        '''
        plt.plot(movement_data[col_tag+ "_x"][movement_data["trial"]==trial],movement_data[col_tag+"_y"][movement_data["trial"]==trial])
        plt.scatter(movement_data[col_tag+ "_x"][movement_data["trial"]==trial][movement_data[col_tag+ "_x"][movement_data["trial"]==trial].index[0]],movement_data[col_tag+"_y"][movement_data["trial"]==trial][movement_data[col_tag+"_y"][movement_data["trial"]==trial].index[0]],marker='x', s=10,color="g") # plot the starting point in red
        plt.scatter(movement_data[col_tag+ "_x"][movement_data["trial"]==trial][movement_data[col_tag+ "_x"][movement_data["trial"]==trial].index[-1]],movement_data[col_tag+"_y"][movement_data["trial"]==trial][movement_data[col_tag+"_y"][movement_data["trial"]==trial].index[-1]],marker='o', s=10,color="r") # plot the starting point in red
        
        plt.title('Trial ' + str(trial)) # plot title
        plt.ylabel('Y-axis')
        plt.xlabel('X-axis')
        plt.xlim(-0.75,0.75)
        plt.ylim(-0.75,0.75)
    

    def filter_movement(self,dataframe,cutoff_frequency=10,fs=1000,output_file=None):
        '''
        The filter_movement function is used to filter the data
        Inputs
        ----------
        dataframe: pandas dataframe
            input dataframe dataframe (9 columns)
            
        cutoff_frequency: int
         lowpasss filter (default: 10)
        
        fs: int
         the sampling frequency (default: 1000)

        Returns
        ----------
        data_extracted: pandas dataframe
            input dataframe dataframe (10 columns), two new columns were added ("axis_filt_x","axis_filt_y")
        '''
        statesSignalX=dataframe["axis_states_x"]; statesSignalY=dataframe["axis_states_y"]
        rawSignalX=dataframe["axis_raw_x"];rawSignalY=dataframe["axis_raw_y"]
        
        # Design the Butterworth filter
        order = 2  # The order of the filter
        b, a = butter(order, cutoff_frequency, btype='low', analog=False,fs=fs)

        # Apply the filter to the signal
        filteredStateSignal_x = filtfilt(b, a, statesSignalX)
        filteredStateSignal_y  = filtfilt(b, a, statesSignalY)
        filteredRawSignal_x = filtfilt(b, a,rawSignalX)
        filteredRawSignal_y  = filtfilt(b, a, rawSignalY)
        
        dataframe["axis_statesfilt_x"]=filteredStateSignal_x
        dataframe["axis_statesfilt_y"]=filteredStateSignal_y
        dataframe["axis_rawfilt_x"]=filteredRawSignal_x
        dataframe["axis_rawfilt_y"]=filteredRawSignal_y
        
        if output_file != None:
            dataframe.to_csv(output_file ,sep=',')
            
        return dataframe
            


    

                       
       