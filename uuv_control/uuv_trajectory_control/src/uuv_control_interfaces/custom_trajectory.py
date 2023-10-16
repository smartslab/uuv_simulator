#!/usr/bin/env python

import numpy as np
import rospy
from uuv_trajectory_generator import trajectory_point
from tf_quaternion.transformations import quaternion_from_euler
from tf_quaternion.transformations import euler_from_quaternion


class custom_trajectory():
    """
    For Unit Tests and similar
    """

    def __init__(self):
        """Class constructor."""
        self.namespace = rospy.get_namespace()
        
        if self.namespace[0] == '/':
            self.namespace= self.namespace.lstrip('/')
        
        #for constant velocity
        self.startpose=np.array([0,0,0])
        self.startrot=np.array([0,0,0])
        self.startquat=quaternion_from_euler(self.startrot[0],self.startrot[1],self.startrot[2])
        self.des_vel=.3
        self.start_time=rospy.get_time()+5.0 
        self.end_time=7.0     
        self.last_time=rospy.get_time()
        self.mark=0
        self.mark2=0
        self.lastq=self.startquat
    
    
    def interpolate3(self,time):
        """For a constant velocity"""
        if time < self.start_time:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        else:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]+(time-self.start_time)*self.des_vel], quat=self.startquat, lin_vel=[0,0,self.des_vel], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        return this_point
        
    def interpolate2(self,time):
        """For a constant velocity that ends at the end time"""
        if time < self.start_time:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        elif time> self.end_time:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0]+(self.end_time-self.start_time)*self.des_vel, self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        else:   
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0]+(time-self.start_time)*self.des_vel, self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[self.des_vel,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        return this_point
        
    def interpolate4(self,time):
        """For a sinusoidal velocity"""
        #rospy.logwarn('interpolating')
        if time < self.start_time:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        else:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0]+30*self.des_vel*np.sin(.5*time), self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,self.des_vel], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        return this_point
        
        
    def interpolate(self,time):
        """For go to a depth at the desired velocity and stay there"""
        depth=-1
        if time < self.start_time:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        elif self.startpose[2]+(time-self.start_time)*self.des_vel < depth:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]+(time-self.start_time)*self.des_vel], quat=self.startquat, lin_vel=[0,0,self.des_vel], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        else:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],depth], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        return this_point


    def interpolate2(self,time):
        """For go to a point at the desired velocity and stay there"""
        depth=-10 #z
        x=10
        y=10
        if time < self.start_time:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        elif self.startpose[2]+(time-self.start_time)*self.des_vel > depth:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0]+(time-self.start_time)*self.des_vel, self.startpose[1]+(time-self.start_time)*self.des_vel,self.startpose[2]+(time-self.start_time)*self.des_vel], quat=self.startquat, lin_vel=[self.des_vel,self.des_vel,self.des_vel], ang_vel=[0,0,0], lin_acc=[0,0,0], ang_acc=[0,0,0])
        else:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[x, y, depth], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
        return this_point
        
    def interpolate1(self,time):
        """Halfway around a pole, then back"""
        #rospy.logwarn(str(self.start_time-time))
        #rospy.logwarn(str(time)+",1692137549.2548687,"+str(self.start_time)+","+str(self.mark)+","+str(self.mark2))
        if time < self.start_time:
            #rospy.logwarn("Waiting")
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[self.startpose[0], self.startpose[1],self.startpose[2]], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
            quatn=self.startquat
        
        elif self.mark==0:
            alpha=-.1 #.0349 #rad/s ~2 deg/sec
            [r,p,y]=[0,0,alpha*(time-self.start_time)]
            quatn=quaternion_from_euler(r,p,y)
            #rospy.logwarn(str(y))
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[2*np.sin(-y), -2+2*np.cos(-y), 0], quat=quatn, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0], ang_acc=[0,0,0])
            if np.abs(1-quatn[2])>.98 and quatn[3]<0:
                self.mark=1

        elif self.mark2==0:
            alpha=.1 #.0349 #rad/s ~2 deg/sec
            [r,p,y]=[0,0,alpha*(time-self.start_time)]
            quatn=quaternion_from_euler(r,p,y)
            #rospy.logwarn(str(y))
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[2*np.sin(-y), -2+2*np.cos(-y), 0], quat=quatn, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0], ang_acc=[0,0,0])
            if np.abs(1-quatn[3])>.98 and quatn[2]<0:
                self.mark2=1
        else:
            this_point=trajectory_point.TrajectoryPoint(t=time, pos=[0, 0, 0], quat=self.startquat, lin_vel=[0,0,0], ang_vel=[0,0,0], lin_acc=[0,0,0],ang_acc=[0,0,0])
            quatn=self.startquat
            
        self.last_time=time
        self.lastq=quatn
        return this_point    
    
