# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pandas as pd
import math


# plots---------------------------
#import matplotlib.pyplot as plt
#import seaborn as sns
# extract fct: 
# plotting
#

class Movement_analyses:
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
            for sess_nb in range(0,len(self.session_names)):
                sess=config["list_subjects"][subject_name]["sess0" + str(sess_nb+1)]
                for run_name in self.runs[subject_name]["sess0" + str(sess_nb+1)]:
                    if not os.path.exists(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/" + run_name +"/ReactionTime/"):
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/" + run_name +"/ReactionTime/") # create reaction time analyses folder
                        os.mkdir(self.data_dir + "/" + subject_name + "/sess_"+ sess + "/" + run_name +"/AngleDev/") # create reaction time analyses folder
                        
        #>>>  read data  -------------------------------------
        self.calib_data={}; # calibration of the participant => position of the different targets
        self.movement_data={}; # Joystick recordings, filter x and y positions
        self.trial_data={}; # info about target orders and trials duration
        
        for subject_name in self.subject_names:
            self.calib_data[subject_name]={};self.movement_data[subject_name]={};self.trial_data[subject_name]={};
            
            for sess_nb in range(0,len(self.session_names)):
                sess=config["list_subjects"][subject_name]["sess0" + str(sess_nb+1)] # session name
                self.movement_data[subject_name][sess]={};self.trial_data[subject_name][sess]={};
                self.calib_data[subject_name][sess]=glob.glob(self.data_dir + "/" + subject_name +  "/sess_" + sess  + "/calib/*calib*cardinals.dat")[-1]
                
                for run_name in self.runs[subject_name]["sess0" + str(sess_nb+1)]:
                    print(run_name)
                    self.movement_data[subject_name][sess][run_name]=glob.glob(self.data_dir + "/" + subject_name +  "/sess_" + sess  + "/" + run_name +"/*movement.csv")[0]
                    self.trial_data[subject_name][sess][run_name]=glob.glob(self.data_dir + "/" + subject_name +  "/sess_" + sess  + "/" + run_name +"/*trial.dat")[0]
        
                                   
   
    def readData(self):
        self.calib_table={};self.movement_table={};self.trial_table={};
        for subject_name in self.subject_names:
            self.calib_table[subject_name]={};self.movement_table[subject_name]={};self.trial_table[subject_name]={};
             
            for sess_nb in range(0,len(self.session_names)):
                sess=self.config["list_subjects"][subject_name]["sess0" + str(sess_nb+1)] # session name
                self.movement_table[subject_name][sess]={};self.trial_table[subject_name][sess]={};
                self.calib_table[subject_name][sess]=pd.read_table(self.calib_data[subject_name][sess],delimiter=" ")
                for run_name in self.runs[subject_name]["sess0" + str(sess_nb+1)]:
                    self.movement_table[subject_name][sess][run_name]=pd.read_table(self.movement_data[subject_name][sess][run_name],delimiter=",")
                    self.trial_table[subject_name][sess][run_name]=pd.read_table(self.trial_data[subject_name][sess][run_name],delimiter=" ")
        
        return self.calib_table, self.movement_table, self.trial_table
    
    def reachAngles(self,calib_table,movement_table,trial_table):
        
        self.homePos=[calib_table["x"].center, calib_table["y"].center] #Home X and Y position for the joystick

        reachAngs = []
        for idx, row in movement_table.iterrows():
            if row.trial>=0 and idx>0:
                Ang2Reach=trial_table["target.angle"][row.trial]
                
                reachAngs.append(self._angleDevCalc(row.axis_rawfilt_x,row.axis_rawfilt_y,Ang2Reach))
                
            else:
                reachAngs.append(np.nan)
   
        movement_table["reachAngs"]=reachAngs

        return movement_table
    
    
    def reachLine(self,calib_table,movement_table,trial_table): 
        
        line2Reach = [];trial_samples_nb=[]
        
        for idx, row in trial_table.iterrows():
            if row.trial==0:
                # target position for initial target (the previous one)
                initTargPosX=calib_table["x"].center
                initTargPosY=calib_table["y"].center
            else:
                initTarg=trial_table["target.angle"][row.trial-1] # Target degree:
                initTargPosX,initTargPosY=self._targetPosition(initTarg,calib_table)# target position
            actualTarg=trial_table["target.angle"][row.trial] # Target degree
            actualTargPosX,actualTargPosY=self._targetPosition(actualTarg,calib_table)
                
            # number of sample for the trial
            trial_sample=len(movement_table[movement_table["trial"]==row.trial])
            trial_sample_nb=range(0,trial_sample)

            # Generate the line points
            line_points = self._linePoints([initTargPosX,initTargPosY],[actualTargPosX,actualTargPosY], trial_sample)
            line2Reach.append(line_points)
            trial_samples_nb.append(trial_sample_nb)
            
        line2ReachConcat=np.concatenate(line2Reach) # concatenate all line2Reach list in an array
        trial_samples=np.concatenate(trial_samples_nb)
        #for idx, row in movement_table.iterrows():
        movement_table["line2Reach_X"]=float('nan') ; movement_table["line2Reach_Y"]=float('nan');movement_table["trial_samples"]=float('nan') # create 2 new columns
        first_trial_index = movement_table[movement_table['trial'] == 0].index[0]
        movement_table["line2Reach_X"][first_trial_index:]=[arr[0] for arr in line2ReachConcat]
        movement_table["line2Reach_Y"][first_trial_index:]=[arr[1] for arr in line2ReachConcat]
        movement_table["trial_samples"][first_trial_index:]=[arr for arr in trial_samples]
      
        lineDistance=[];lineDev=[]
        for idx, row in movement_table.iterrows():
            if row.trial>=0 and idx>0:
                reachAngs=trial_table["target.angle"][row.trial]
            else:
                reachAngs=[np.nan]
            
            distance,dev=self._lineDev(movement_table["axis_raw_x"][idx],movement_table["axis_raw_y"][idx],movement_table["line2Reach_X"][idx],movement_table["line2Reach_Y"][idx],reachAngs)
            lineDistance.append(distance)
            lineDev.append(dev)
            
        movement_table["lineDistance"]=[arr[0] for arr in lineDistance]
        movement_table["lineDev"]=[arr for arr in lineDev]
      
            
        
        return movement_table
                
    def _kin_charsigned(trial_samples, vel_kin, t, tardeg):
        '''
        This program function calculate some kinematic characteristics, signed area, Perpendicular Deviation, Path Length and Initial Angular Deviation

        Attributes
        ----------
        trial_samples : array
            array that contained movement recordings with the following shape: (x coordinates, y coordinates, number of samples)
        
        vel_kin 
        
        t: array 
           time of each point recording
           
        tardeg: int
            value of the target position in degree
        
        '''
        
        t = t - t[0] # substract each time by the initial one to start with t=0

        L = trial_samples.shape[0]  # sample size
        strx = trial_samples[0, 0]  # start_point, x coordinates
        stry = trial_samples[0, 1]  # start_point, y coordinates
       
        stpx = trial_samples[L, 0]   # stop_point, x coordinates
        stpy = trial_samples[L, 1]  # stop_point, y coordinates

        area = np.zeros(L) # empty array for signed area
        pd = np.zeros(L) # empty array for perpendicular deviation
        dd = np.zeros(L) # 
        iad = np.zeros(L) # empty array for initial angular deviation
        lng = np.zeros(L) #

        for i in range(1, L): # 
            datax = trial_samples[i, 0]  # Each point of data
            datay = trial_samples[i, 1]  # Each point of data
            datax1 = trial_samples[i - 1, 0]  # Previous point of data
            datay1 = trial_samples[i - 1, 1]  # Previous point of data
            vector1 = np.array([datax - strx, datay - stry, 0]) / np.sqrt(np.sum((np.array([datax - strx, datay - stry]) ** 2)))
            vector2 = np.array([stpx, stpy, 0]) / np.sqrt(np.sum((np.array([stpx, stpy]) ** 2)))
            theta = np.abs(np.rad2deg(np.arccos(np.dot(vector1, vector2))))

            if np.sum(np.cross(vector1, vector2)) < 0:
                theta = -theta

            iad[i] = theta  # Initial angular deviation
            lng[i] = np.sqrt(np.sum((np.array([datax - datax1, datay - datay1]) ** 2)))  # distance between two points
            pd[i] = np.sin(np.deg2rad(theta)) * np.sqrt(np.sum((np.array([datax - strx, datay - stry]) ** 2)))  # Perpendicular distance
            dd[i] = np.cos(np.deg2rad(theta)) * np.sqrt(np.sum((np.array([datax - strx, datay - stry]) ** 2)))  # directional distance
            area[i] = (pd[i] + pd[i - 1]) * (dd[i] - dd[i - 1]) / 2

        place = np.argmax(np.sqrt(vel_kin[:, 0] ** 2 + vel_kin[:, 1] ** 2))

        t200 = np.argmin(np.abs(t - 0.2))[0]
        t100 = np.argmin(np.abs(t - 0.1))[0]

        t_area = np.sum(area)
        t_pdmaxv = pd[place]
        t_pd200 = pd[t200]
        t_pd100 = pd[t100]
        t_pdend = pd[-1]
        _, dummy = np.argmax(np.abs(pd))
        t_pd = pd[dummy]

        t_lng = np.sum(lng)
        t_iadmaxv = iad[place]  # The initial deviation, at the maximum tangential velocity
        t_iad200 = iad[t200]   # The initial deviation, 200ms into the movement
        t_iad100 = iad[t100]   # The initial deviation, 100ms into the movement
        t_iadend = iad[-1]     # The initial deviation, end of the movement
        t_meanpd = np.mean(pd)

        return t_meanpd, t_area, t_pd, t_pdmaxv, t_pd100, t_pd200, t_pdend, t_lng, t_iadmaxv, t_iad100, t_iad200, t_iadend

        #return angular devition for a signle raw
        def _angleDevCalc(self,cursorPosX,cursorPosY, Ang2Reach):
            return math.degrees(math.atan2(cursorPosY - self.homePos[1], cursorPosX - self.homePos[0]))-Ang2Reach

        # return to target position     
        def _targetPosition(self,targetAngleDegree,calibTable):
            #targetAngle=self.trial_table[subject_name][sess][run_name]["target.angle"][trial] # position of the target to reach
            if targetAngleDegree==90:
                target_posX=calibTable["x"].front
                target_posY=calibTable["y"].front
            elif targetAngleDegree==0:
                target_posX=calibTable["x"].right
                target_posY=calibTable["y"].right
            elif targetAngleDegree==-90:
                target_posX=calibTable["x"].back
                target_posY=calibTable["y"].back
            elif targetAngleDegree==180:
                target_posX=calibTable["x"].left
                target_posY=calibTable["y"].left


            return target_posX, target_posY
   
    def _lineDev(self,posX,posY,line2reachX,line2reachY,Ang2Reach):
        
        # Calculate the x and y increments between the line2reach and the actual movement
        delta_x = posX - line2reachX
        delta_y = posY - line2reachY
            
        lineDistance=[abs(delta_x),abs(delta_y)] # x and y distance between the two lines
        lineDev=abs(math.degrees(math.atan2(posX, posY))-math.degrees(math.atan2(line2reachX, line2reachY)))# deviation in degree

        
        return lineDistance,lineDev
    
    def _linePoints(self,posStart,posEnd, sampleSize):
        # Calculate the x and y increments
        delta_x = (posEnd[0] - posStart[0]) / (sampleSize - 1)
        delta_y = (posEnd[1] - posStart[1]) / (sampleSize - 1)

        # Generate the line points
        line_points = [(posStart[0] + delta_x * i, posStart[1] + delta_y * i) for i in range(sampleSize)]

        return line_points

        
        
    def reaction_time(self,subject_name, data_filename=False):
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

        # read trials files
         # trials files --------------------------------------
        1+1
        
        