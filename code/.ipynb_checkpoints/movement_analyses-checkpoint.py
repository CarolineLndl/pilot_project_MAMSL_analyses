# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pandas as pd
import math
from scipy.stats import linregress

import matplotlib.pyplot as plt

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
                    self.trial_data[subject_name][sess][run_name]=glob.glob(self.data_dir + "/" + subject_name +  "/sess_" + sess  + "/" + run_name +"/*trialfilter.csv")[0]
        
                                   
   
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
                    self.trial_table[subject_name][sess][run_name]=pd.read_table(self.trial_data[subject_name][sess][run_name],delimiter=",")
        
        return self.calib_table, self.movement_table, self.trial_table
    
    def kinematics(self,movement_table_init,trial_table_init,run_name, angDev=True,output_dir=None):
        '''
        Compute different kinematics
        
        Attributes
        ----------
        movement_table : dict
        dictonnary with information about each data points (one line per sample)
        
        trial_table: dict
        dictonnary with information about each trial (one line per trial)
        '''
        # I. Prepare the dataframe that will store the data
        # -------------------------------------------------------------------------
        # I.a Copy original dataframe
        movement_table= movement_table_init.copy()
        trial_table= trial_table_init.copy()
        
        #I.b Crop the table for additional lines before trial 0
        movement_table = movement_table[movement_table["trial"] >= 0][1:] # do not considere first rows

        #I.c Add some information in the dataframe
        sequence = [];trial_table["seq_nb"]=None;trial_table["blockRun"]=None;trial_table["seqRun"]=None
        trial_samples_nb=[] # create empty dataframe
        movement_table["trial_samples"]=0 # create  new column with 0 values
            
        for idx, row in trial_table.iterrows():
            #Report the number of sample for the trial
            trial_sample=len(movement_table[movement_table["trial"]==row.trial])
            trial_sample_nb=range(0,trial_sample)
            trial_samples_nb.append(trial_sample_nb)
            
            # Report information about the sequence number
            number = idx
            sequence.extend([number] * 8)
            number += 1
            trial_table.loc[idx,"seq_nb"]=sequence[idx]
            
        trial_samples=np.concatenate(trial_samples_nb) # catenate the values
        movement_table.loc[:,"trial_samples"]=[arr for arr in trial_samples] # add trial simple info in the main table
 
        # Report information about block number including all runs
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
        
        # Select specific columns for the output               
        movement_table=movement_table[["sample","trial","trial_samples","t_since_start","axis_rotfilt_x","axis_rotfilt_y","straightened_mvmnt_X","straightened_mvmnt_Y","straightened_rotmvmnt_X","straightened_rotmvmnt_Y"]] # select subset of the main table
        trial_table=trial_table[["subject","run_name","block","blockRun","seqRun","seq_nb","trial","remove_trial","remove_seq","seqInBlock","trialInSeq","target.angle","t.move","t.hit"]] # select subset of the main table
        
        #II. Analyse movement kinematics
        # -------------------------------------------------------------------------
        #B. Calculate movement velocity
        movement_table.loc[:,"velocity"]=float('nan');
        velocity=[]
        for idx, row in trial_table.iterrows():
            vel = self._velocity(movement_table["straightened_mvmnt_X"][movement_table["trial"]==row.trial],movement_table["straightened_mvmnt_Y"][movement_table["trial"]==row.trial])
            velocity.append(vel)
        velocityConcat=np.hstack(velocity).reshape(-1, 1) # concatenate all trials
        movement_table.loc[:,"velocity"]=velocityConcat # copy the results in the main dataframe

        #C. Calculate angular deviation
        if angDev==True:
            movement_table,trial_table=self._reachLine(movement_table,trial_table,run_name,line2reach="straightLine")

                  
        #D. save results
        if output_dir != None:
            movement_table.to_csv(output_dir+'movement_table.csv', index=False)
            trial_table.to_csv(output_dir+'trial_table.csv', index=False)
        
        return movement_table,trial_table

    def _velocity(self,posX,posY):
        '''
        Calculate the velocity of the movement
        
        Attributes
        ----------
        posX: string
            x coordinates of the movement
        posY: string
            y coordinates of the movement
        
        '''
        
        vel=np.sqrt(np.diff(posX) ** 2 + np.diff(posX)** 2)
        vel= np.insert(vel, 0, 0)
        
        return vel
        
    def _reachLine(self,movement_table,trial_table,run_name,line2reach="straightLine"):
        '''
        Calculate the shorter line between the start and the end of the movement
        
        Attributes
        ----------
        line2reach: string
        "mvmntPos": will be calculated between the position of the first and last sample of the trial
        "targetPos": will be calculated between the position of the initial target and the target to reach 
        
        '''
        
        line2Reach = [];lineVel=[]
        # I. Calculate the line to reach --------------------------------------------------------------------
        for idx, row in trial_table.iterrows():
            initTargPosX=0; initTargPosY=0 # position of the reference line
            actualTargPosX=0 # starting position of the movement
            if row.seqInBlock==0 and row.trialInSeq==0:
                actualTargPosY=0.5 # for shorter trials
            else:
                actualTargPosY=0.9
 
            # number of sample for the trial
            trial_sample=len(movement_table[movement_table["trial"]==row.trial])
            trial_sample_nb=range(0,trial_sample)
            
            # Generate the line points
            line_points = self._linePoints([initTargPosX,initTargPosY],[actualTargPosX,actualTargPosY], trial_sample)
            line2Reach.append(line_points)

            # Calculate the line between the initial joystik position and final position
            iniLoc=movement_table[(movement_table["trial"]==row.trial)].index[0]# initial point
            endLoc=movement_table[(movement_table["trial"]==row.trial)].index[-1]# final point

            initPosX=movement_table["straightened_mvmnt_X"][iniLoc];initPosY=movement_table["straightened_mvmnt_Y"][iniLoc]
            endPosX=movement_table["straightened_mvmnt_X"][endLoc]; endPosY=movement_table["straightened_mvmnt_Y"][endLoc]

            line_points = self._linePoints([initPosX,initPosY],[endPosX,endPosY], trial_sample)
            line2ReachConcat=np.concatenate(line2Reach) # concatenate all line2Reach list in an array
            

        #Copy the results in the main table
        movement_table.loc[:,"line2Reach_X"]=float('nan') ; movement_table.loc[:,"line2Reach_Y"]=float('nan');
              
        movement_table.loc[:, "line2Reach_X"] = [arr[0] for arr in line2ReachConcat]
        movement_table.loc[:, "line2Reach_Y"] = [arr[1] for arr in line2ReachConcat]
        
        # II. Calculate the deviation between the actual movement and the line to reach ---------------------------------------------------
        lineDistance=[];lineDev=[]
        for idx, row in trial_table.iterrows():
            angDev=self._lineDev(movement_table["straightened_mvmnt_X"][movement_table["trial"]==row.trial],movement_table["straightened_mvmnt_Y"][movement_table["trial"]==row.trial],movement_table["line2Reach_X"][movement_table["trial"]==row.trial],movement_table["line2Reach_Y"][movement_table["trial"]==row.trial])
            lineDev.append(angDev)

        lineDevConcat=np.hstack(lineDev).reshape(-1, 1)

        movement_table.loc[:,"lineDistance"]=float('nan') ; movement_table.loc[:,"lineDev"]=float('nan');
        movement_table.loc[:,"lineDev"]=lineDevConcat
        
        # II. Extract the deviation for the max velocity ---------------------------------------------------
        #trial_table.loc[:,"lineDistance_vel"]=float('nan');trial_table.loc[:,"lineDistance"]=float('nan')
        trial_table.loc[:,"lineDev_vel"]=float('nan');trial_table.loc[:,"lineDev"]=float('nan');
        
        # Extract the maximal value (for the first 100 points) for each trial
        for idx, row in trial_table.iterrows(): # trial number
            if row.remove_trial==0:
                # take the value at the max velocity
                perc5=np.round(len(movement_table["velocity"][(movement_table["trial"]==row.trial)])/5,0) # 5 first percent of the movement
               
                maxVelLoc=movement_table["velocity"][(movement_table["trial"]==row.trial)][int(perc5):].idxmax()
                trial_table.loc[trial_table["trial"]==row.trial,"lineDev_vel"]=movement_table[(movement_table["trial"]==row.trial)]["lineDev"][maxVelLoc]
                trial_table.loc[trial_table["trial"]==row.trial,"lineDev"]=movement_table[(movement_table["trial"]==row.trial)]["lineDev"].index[-1]
            

        return movement_table,trial_table
            

    def _lineDev(self,posX,posY,line2reachX,line2reachY):
        '''
        Calculate the angular deviation between two lines:
        line1 is the line to reach (angle1 =0); line2 is the movement of the participant with variable angles (angle2=a).
        Angular deviation is the difference between the two angles: a-0=deviation of the movement from the 0 line
        Inputs
        ----------
        posX: array
            x position of hand movement
        posY: array
            y position of hand movement
        line2reachX: array
            x position of the reference line (in our specific case it is 0)
        line2reachY: array
            y position of the reference line

        Returns
        ----------
        angular_deviations: angular deviation between the two lines in degree

        '''
        angular_deviations=[] # create an empty array
        for x1, y1, x2, y2 in zip(line2reachX, line2reachY, posX, posY):
            angle1 = math.atan2(x1,y1) # Calculate the angles of the lines using arctangent
            angle2 = math.atan2(x2,y2)# Calculate the angles of the lines using arctangent
            angular_deviation = np.degrees((angle2 - angle1)) #Calculate the angular deviation between the two angles and Convert the angular deviation from radians to degrees
            angular_deviations.append(angular_deviation)


        return angular_deviations

    def _linePoints(self,posStart,posEnd, sampleSize):
        '''
        Create straight lines using initial and final position  along with sample information
        ----------
        posStart: array
            [x,y] position of the initial point
        posEnd: array
            [x,y] position of the final point
        sampleSize: int
            number of sample to be used to create the line
       
        Returns
        ----------
        line_points: list
            tuples representing the x and y coordinates of each point on the line.
            
        '''
        # Calculate the x and y increments
        delta_x = (posEnd[0] - posStart[0]) / (sampleSize - 1)
        delta_y = (posEnd[1] - posStart[1]) / (sampleSize - 1)

        # Generate the line points by iterating over the sample indices (i)
        line_points = [(posStart[0] + delta_x * i, posStart[1] + delta_y * i) for i in range(sampleSize)]

        return line_points
