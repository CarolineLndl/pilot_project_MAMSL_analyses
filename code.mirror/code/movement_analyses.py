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
            for sess_nb in range(0,len(self.config["session_nb"])):
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
            
            for sess_nb in range(0,len(self.config["session_nb"])):
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
             
            for sess_nb in range(0,len(self.config["session_nb"])):
                sess=self.config["list_subjects"][subject_name]["sess0" + str(sess_nb+1)] # session name
                self.movement_table[subject_name][sess]={};self.trial_table[subject_name][sess]={};
                self.calib_table[subject_name][sess]=pd.read_table(self.calib_data[subject_name][sess],delimiter=" ")
                for run_name in self.runs[subject_name]["sess0" + str(sess_nb+1)]:
                    self.movement_table[subject_name][sess][run_name]=pd.read_table(self.movement_data[subject_name][sess][run_name],delimiter=",")
                    self.trial_table[subject_name][sess][run_name]=pd.read_table(self.trial_data[subject_name][sess][run_name],delimiter=" ")
        
        return self.calib_table, self.movement_table, self.trial_table
    
    def kinematics(self,movement_table_init,trial_table_init,run_name, reachLine=True,kinematics=True):
        '''
        Compute different kinematics
        
        Attributes
        ----------
        movement_table : dict
        dictonnary with information about each data points (one line per sample)
        
        trial_table: dict
        dictonnary with information about each trial (one line per trial)
        '''
        # I. Add a column with a value for each sample of each trial starting from 0
        # -------------------------------------------------------------------------
        # Copy original dataframe
        movement_table= movement_table_init.copy()
        trial_table= trial_table_init.copy()
            
        trial_samples_nb=[]
        
        #Crop the table for additional lines before trial 0
        movement_table = movement_table[movement_table["trial"] >= 0][1:] # do not considere first raws

        # a. number of sample for the trial
        sequence = [];trial_table["seq_nb"]=None;trial_table["blockRun"]=None;trial_table["seqRun"]=None
        for idx, row in trial_table.iterrows():
            
            trial_sample=len(movement_table[movement_table["trial"]==row.trial])
            trial_sample_nb=range(0,trial_sample)
            trial_samples_nb.append(trial_sample_nb)
            
            # add information about the sequence number
            number = idx
            sequence.extend([number] * 8)
            number += 1
            trial_table.loc[idx,"seq_nb"]=sequence[idx]
        
        # add information about block number including all runs
        if run_name=="RNDpre":
            trial_table.loc[:,"blockRun"]= trial_table["block"]
            trial_table.loc[:,"seqRun"]= trial_table["seq_nb"]
            
        elif run_name=="RNDpost":
            trial_table.loc[:,"blockRun"]= trial_table["block"] + 4 + 25
            trial_table.loc[:,"seqRun"]= trial_table["seq_nb"] + (4 + 25) *10
        else:
            trial_table.loc[:,"blockRun"]= trial_table["block"] + 4
            trial_table.loc[:,"seqRun"]= trial_table["seq_nb"] + (4) *10

                
        trial_table["run_name"]=run_name
        trial_table=trial_table[["subject","run_name","block","blockRun","seqRun","seq_nb","trial","target.angle","t.move","t.hit"]] # select subset of the main table
    
    
        trial_samples=np.concatenate(trial_samples_nb) # catenate the values
        movement_table["trial_samples"]=0 # create  new column with 0 values
        
        movement_table.loc[:,"trial_samples"]=[arr for arr in trial_samples] # add trial simple info in the main table
        
        movement_table=movement_table[["sample","trial","trial_samples","t_since_start","axis_statesfilt_x","axis_statesfilt_y"]] # select subset of the main table
        
        
        #b. calculate the velocity power
        movement_table.loc[:,"velocity"]=0; movement_table.loc[:,"velocityPow"]=0 ;
       
        velPower = np.sqrt(pow(np.gradient(movement_table["axis_statesfilt_x"]), 2) + pow(np.gradient(movement_table["axis_statesfilt_y"]), 2))
        vel=np.sqrt((np.gradient(movement_table["axis_statesfilt_x"])) + np.gradient(movement_table["axis_statesfilt_y"]))
        movement_table.loc[:,"velocity"]=vel;movement_table.loc[:,"velocityPow"]=velPower
        
        
        if reachLine==True:
            movement_table,trial_table=self._reachLine(movement_table,trial_table,line2reach="mvmntPos")

            
        if kinematics==True:
            movement_table,trial_table=self._kinematic(movement_table,trial_table)
        
        return movement_table,trial_table

    
    def _reachLine(self,movement_table,trial_table,line2reach="mvmntPos"):
        '''
        Calculate the shorter line between the start and the end of the movement
        
        Attributes
        ----------
        line2reach: string
        "mvmntPos": will be calculated between the position of the first and last sample of the trial
        "targetPos": will be calculated between the position of the initial target and the target to reach 
        
        '''
        
        line2Reach = [];
        # I. Calculate the line to reach --------------------------------------------------------------------
        for idx, row in trial_table.iterrows():
            
            #Case1: The line will be drawn beween 2 targets position
            if line2reach=="targetPos":
                if row.trial==0:
                    # target position for initial target (the previous one)
                    initTargPosX=0
                    initTargPosY=0
                
                else:
                    initTarg=trial_table["target.angle"][row.trial-1] # Target degree:
                    initTargPosX,initTargPosY=self._targetPosition(initTarg)# target position
                actualTarg=trial_table["target.angle"][row.trial] # Target degree
                actualTargPosX,actualTargPosY=self._targetPosition(actualTarg)
                          
            # Case2: the line will be drawn beween initial and final movement position
            if line2reach=="mvmntPos":
                index_init=movement_table[movement_table["trial"]==trial_table["trial"][idx]].index[0]
                initTargPosX=movement_table["axis_statesfilt_x"][index_init]
                initTargPosY=movement_table["axis_statesfilt_y"][index_init]
                index_actual=movement_table[movement_table["trial"]==trial_table["trial"][idx]].index[-1]
                actualTargPosX=movement_table["axis_statesfilt_x"][index_actual]
                actualTargPosY=movement_table["axis_statesfilt_y"][index_actual]
                      
                
            # number of sample for the trial
            trial_sample=len(movement_table[movement_table["trial"]==row.trial])
            trial_sample_nb=range(0,trial_sample)
            
            # Generate the line points
            line_points = self._linePoints([initTargPosX,initTargPosY],[actualTargPosX,actualTargPosY], trial_sample)
            line2Reach.append(line_points)
 
        line2ReachConcat=np.concatenate(line2Reach) # concatenate all line2Reach list in an array
        
        #Copy the results in the main table
        movement_table.loc[:,"line2Reach_X"]=float('nan') ; movement_table.loc[:,"line2Reach_Y"]=float('nan');
        movement_table.loc[:, "line2Reach_X"] = [arr[0] for arr in line2ReachConcat]
        movement_table.loc[:, "line2Reach_Y"] = [arr[1] for arr in line2ReachConcat]

       
        
        # II. Calculate the deviation between the actual movement and theline to reach ---------------------------------------------------
        lineDistance=[];lineDev=[]
        for idx, row in movement_table.iterrows():
            if row.trial>=0 and idx>0:
                reachAngs=trial_table["target.angle"][row.trial]
            else:
                reachAngs=[np.nan]
            
            distance,dev=self._lineDev(movement_table["axis_statesfilt_x"][idx],movement_table["axis_statesfilt_y"][idx],movement_table["line2Reach_X"][idx],movement_table["line2Reach_Y"][idx],reachAngs)
            lineDistance.append(distance)
            lineDev.append(dev)
        movement_table.loc[:,"lineDistance"]=float('nan') ; movement_table.loc[:,"lineDev"]=float('nan');
        movement_table.loc[:,"lineDistance"]=[arr[0] for arr in lineDistance]
        movement_table.loc[:,"lineDev"]=[arr for arr in lineDev]
        
        # II. Extract the deviation for the max velocity ---------------------------------------------------
        trial_table.loc[:,"lineDistance"]=float('nan')
        trial_table.loc[:,"lineDev"]=float('nan')
        
        # Extract the maximal value (for the first 100 points) for each trial
        value="d100"
        for idx, row in trial_table.iterrows(): # trial number
            # take the value at the max velocity
            if value=="velocity":
                maxVel=np.max(movement_table["velocityPow"][(movement_table["trial"]==row.trial)])
                maxVelLoc=movement_table[(movement_table["trial"]==row.trial)].loc[movement_table["velocityPow"]==maxVel].index[0]                  
                trial_table.loc[trial_table["trial"]==row.trial,"lineDistance"]=movement_table["lineDistance"][maxVelLoc]
                trial_table.loc[trial_table["trial"]==row.trial,"lineDev"]=movement_table["lineDev"][maxVelLoc]
                
            # take the mean value for the first 100 points
            if value=="d100":
                trial_table.loc[trial_table["trial"]==row.trial,"lineDistance"]=np.mean(movement_table[(movement_table["trial"]==row.trial) & (movement_table["trial_samples"]<50)]["lineDistance"])
                trial_table.loc[trial_table["trial"]==row.trial,"lineDev"]=np.mean(movement_table[(movement_table["trial"]==row.trial) & (movement_table["trial_samples"]<50)]["lineDev"])


        return movement_table, trial_table
    
    
        
        
    def _kinematic(self,movement_table,trial_table): 
        
        perpDist_all=[];dirDist_all=[];area_all=[]
        for idx, row in trial_table.iterrows():
            ang2Reach=trial_table["target.angle"][row.trial]# extract target position in degree 
            axisX=np.array(movement_table["axis_statesfilt_x"][movement_table["trial"]==row.trial])
            axisY=np.array(movement_table["axis_statesfilt_x"][movement_table["trial"]==row.trial])
            acqTime=np.array(movement_table["t_since_start"][movement_table["trial"]==row.trial])
            perpDist,dirDist,area =self._kin_charsigned(axisX,axisY,acqTime,ang2Reach)
            
            perpDist_all.append(perpDist); dirDist_all.append(dirDist);area_all.append(area)
            
                 
        perpDistConcat=np.hstack(perpDist_all)#[0] # concatenate all AngDev_all list in an array
        dirDistConcat=np.hstack(dirDist_all);areaConcat=np.hstack(area_all)
        movement_table.loc[:,"perpDist"]=float('nan');movement_table.loc[:,"dirDist"]=float('nan') ;movement_table.loc[:,"area"]=float('nan') 
          
        movement_table.loc[:,"perpDist"]=[arr for arr in perpDistConcat];movement_table.loc[:,"dirDist"]=[arr for arr in dirDistConcat]
        movement_table.loc[:,"area"]=[arr for arr in areaConcat]

         # Extract the maximal value at the max velocity
        trial_table.loc[:,"perpDist"]=float('nan');trial_table.loc[:,"dirDist"]=float('nan');trial_table.loc[:,"area"]=float('nan')
        value="d100"
        for idx, row in trial_table.iterrows(): # trial number
            if value=="velocity":
                maxVel=np.max(movement_table["velocityPow"][(movement_table["trial"]==row.trial)])
                maxVelLoc=movement_table[(movement_table["trial"]==row.trial)].loc[movement_table["velocityPow"]==maxVel].index[0]                  
                trial_table.loc[trial_table["trial"]==row.trial,"perpDist"]=movement_table["perpDist"][maxVelLoc]
                trial_table.loc[trial_table["trial"]==row.trial,"dirDist"]=movement_table["dirDist"][maxVelLoc]
                trial_table.loc[trial_table["trial"]==row.trial,"area"]=movement_table["area"][maxVelLoc]
            if value=="d100":
                trial_table.loc[trial_table["trial"]==row.trial,"perpDist"]=np.mean(movement_table[(movement_table["trial"]==row.trial) & (movement_table["trial_samples"]<100)]["perpDist"])
                trial_table.loc[trial_table["trial"]==row.trial,"dirDist"]=np.mean(movement_table[(movement_table["trial"]==row.trial) & (movement_table["trial_samples"]<100)]["dirDist"])
                trial_table.loc[trial_table["trial"]==row.trial,"area"]=np.mean(movement_table[(movement_table["trial"]==row.trial) & (movement_table["trial_samples"]<100)]["area"])

            
        return movement_table,trial_table
            
                             
               

                
    def _kin_charsigned(self,axisX,axisY,t,tardeg):
        '''
        This program function calculate some kinematic characteristics, signed area, Perpendicular Deviation, Path Length and Initial Angular Deviation

        Attributes
        ----------
        trial_samples : array
            array that contained movement recordings with the following shape: (x coordinates, y coordinates, number of samples)
        
                
              
        '''
        
        t = t - t[0] # substract each time by the initial one to start with t=0

        L = len(axisX)  # sample size
        strx = axisX[0] # start_point, x coordinates
        stry = axisY[0]  # start_point, y coordinates
       
        stpx = axisX[-1]   # stop_point, x coordinates
        stpy = axisY[-1]  # stop_point, y coordinates

        area = np.zeros(L) # empty array for signed area
        pd = np.zeros(L) # empty array for perpendicular deviation
        dd = np.zeros(L) # 
        iad = np.zeros(L) # empty array for initial angular deviation
        lng = np.zeros(L) #empty array distance between two points

        for i in range(1, L): # 
            datax = axisX[i]  # Each point of data
            datay = axisY[i]  # Each point of data
            datax1 = axisX[i - 1]  # Previous point of data
            datay1 = axisY[i - 1]  # Previous point of data
            
            vector1 = np.array([datax - strx, datay - stry, 0]) / np.sqrt(np.sum((np.array([datax - strx, datay - stry]) ** 2)))
            vector2 = np.array([stpx, stpy, 0]) / np.sqrt(np.sum((np.array([stpx, stpy]) ** 2)))
            
            #theta = np.abs(np.rad2deg(np.arccos(np.dot(vector1, vector2)))) # position in degree
            theta = np.dot(vector1, vector2); theta = np.arccos(np.clip(theta, -1.0, 1.0))
            theta = np.abs(np.rad2deg(theta))
            if np.sum(np.cross(vector1, vector2)) < 0:
                theta = -theta

            #iad[i] = theta  # Initial angular deviation
            lng[i] = np.sqrt(np.sum((np.array([datax - datax1, datay - datay1]) ** 2)))  # distance between two points
            pd[i] = np.sin(np.deg2rad(theta)) * np.sqrt(np.sum((np.array([datax - strx, datay - stry]) ** 2)))  # Perpendicular distance
            dd[i] = np.cos(np.deg2rad(theta)) * np.sqrt(np.sum((np.array([datax - strx, datay - stry]) ** 2)))  # directional distance
            area[i] = (pd[i] + pd[i - 1]) * (dd[i] - dd[i - 1]) / 2

        #place = np.argmax(np.sqrt(vel_kin[:, 0] ** 2 + vel_kin[:, 1] ** 2))
        t200 =np.argmin(np.abs(t - 0.2)) #t200 = np.argmin(np.abs(t - 0.2))[0]
        t100 = np.argmin(np.abs(t - 0.1))#np.argmin(np.abs(t - 0.1))[0]

        t_area = np.sum(area)
        #t_pdmaxv = pd[place]
        t_pd200 = pd[t200]
        t_pd100 = pd[t100]
        t_pdend = pd[-1]
        dummy = np.argmax(np.abs(pd))
        t_pd = pd[dummy]

        t_lng = np.sum(lng)
        #t_iadmaxv = iad[place]  # The initial deviation, at the maximum tangential velocity
        t_iad200 = iad[t200]   # The initial deviation, 200ms into the movement
        t_iad100 = iad[t100]   # The initial deviation, 100ms into the movement
        t_iadend = iad[-1]     # The initial deviation, end of the movement
        t_meanpd = np.mean(pd)

        return abs(pd),abs(dd),abs(area)
    
    #return angular devition for a signle raw
    def _angleDevCalc(self,cursorPosX,cursorPosY, Ang2Reach):
        return math.degrees(math.atan2(cursorPosY - self.homePos[1], cursorPosX - self.homePos[0]))-Ang2Reach

    # return to target position     
    def _targetPosition(self,targetAngleDegree):
        radius = .50
        target_posX=radius * np.cos(np.pi * targetAngleDegree / 180)
        target_posY=radius * np.sin(np.pi * targetAngleDegree / 180)
       

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

