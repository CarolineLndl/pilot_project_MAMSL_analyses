# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# plots---------------------------
#import matplotlib.pyplot as plt
#import seaborn as sns
# extract fct: 
# plotting
#

class Movement_plots:
    '''
    Plot movements
    Attributes
    ----------
    config : dict
    '''
    
    def __init__(self,movement_table,trial_table):
                
        self.movement_data=movement_table
        self.trial_data=trial_table
        
                                   
    def plot_1d(self,trial_range=None,plot_arc=True,plot_targets=True,cmap_color="jet"):
        
        
        if trial_range ==None:
            trial_range=range(0,len(self.trial_data))
        
        self.cmap=cm.get_cmap(cmap_color, len(trial_range)) 
        # 1. Create the empty figure
        fig, ax = plt.subplots()
        
        # 2. Plot the rotation arc
        if plot_arc==True:
            # Parameters for the arc
            arc_centre = (0, 0)
            arc_radius = 1.2
            start_angle = 60  # in degrees
            end_angle = 90  # in degrees
            
            # Create the arc as a Patch
            arc = patches.Arc(arc_centre, 2*arc_radius, 2*arc_radius, angle=0, theta1=start_angle, theta2=end_angle, color='black',linewidth=2)
            
            # Add the arc to the axis
            ax.add_patch(arc)
            # Set equal aspect ratio
            ax.set_aspect('equal')
            ax.text(arc_centre[0] + arc_radius * np.cos(np.radians(90)), arc_centre[1] + arc_radius * np.sin(np.radians(90))+0.1, '0', ha='center', va='center')
            ax.text(arc_centre[0] + arc_radius * np.cos(np.radians(60)), arc_centre[1] + arc_radius * np.sin(np.radians(60))+0.1, '30', ha='center', va='center')
            
        # 3. Plot target positions
        if plot_targets==True:
            # Parameters for the targets
            targ_centre = [0, 0]
            targetdirections = [0, 90, -90, 180]
            targ_radius = .50
            
            # Add the target to the axis
            ax.plot((targ_radius * np.cos(np.pi * 90 / 180)), (targ_radius * np.sin(np.pi * (90) / 180)+0.4), color="gray",marker="o", linestyle="None", markersize=20,alpha=0.5)
            ax.plot(targ_centre[0], targ_centre[1], marker="X", linestyle="None", markersize=10,alpha=0.5)
        
        # 3. Plot movement     
        for trial in trial_range:
                sub=self.movement_data[self.movement_data["trial"]==trial]
                color=self.cmap(trial)
                ax.plot(sub["straightened_mvmnt_X"],sub["straightened_mvmnt_Y"],color=color)
              
        #define graph limits
        plt.xlim(-0.8,0.8)
        plt.ylim(-0.2,1.5)

            

        
  
        
