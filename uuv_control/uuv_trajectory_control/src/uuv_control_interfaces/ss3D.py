#!/usr/bin/env python

import numpy as np
import rospy

class ss3D(object):
    """
    Creates a SISO state space system with three states for use in the noise estimator
    """
    
    def __init__(self, a0, a1, a2, a3, a4, a5, a6, a7, a8, b0, b1, b2, c0, c1, c2):
        self.A=np.array([[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]])
        self.B=np.array([[b0],[b1],[b2]])
        self.C=np.array([[c0,c1,c2]])
        
        self.x=np.array([[0,0,0]])
        self.x=self.x.transpose()
        
        self.xd=np.array([[0,0,0]])
        self.xd=self.xd.transpose()
        
        self.y=0
        self.t=0
        
    def update_sys(self, u):
        self.xd=np.dot(self.A,self.x)+self.B*u
        self.y=np.dot(self.C,self.x)
        tnew=rospy.get_time()
        tdiff=tnew-self.t
        self.x=self.x+self.xd*tdiff
        self.t=tnew
        return self.y
