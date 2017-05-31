
# Filename: quadcopter.py
# Description: simulate the quadcopter dynmaics with inverted pendulum

class quadcopter(object):
    """docstring for quadcopter"""
    def __init__(self, time_step):
        self.t = time_step
        # parameters
        self.gravity = 9.81;
        self.Ts = 0.01;
        #physical parameters of airframe
        self.l = 0.2; # m, Distance between rotor and center
        self.k1 = 100; # propellers constant
        self.k2 = 100; # propellers constant 
        self.R = 0.04; # m, Center mass radius 
        self.M = 1 # kg, Body weight
        self.m = 0.07 #kg, Rotor weight 
        self.mass = self.M + self.m;
        self.Jx   = 2*self.M*self.R^2/5 + 2*self.l*self.m;
        self.Jy   = 2*self.M*self.R^2/5 + 2*self.l*self.m;
        self.Jz   = 2*self.M*self.R^2/5 + 4*self.l*self.m;
        self.Jxz  = 0;
        # initial conditions
        self.pn0    = 0;  # initial North position
        self.pe0    = 0;  # initial East position
        self.pd0    = 0;  # initial Down position (negative altitude)
        self.u0     = 0;  # initial velocity along body x-axis
        self.v0     = 0;  # initial velocity along body y-axis
        self.w0     = 0;  # initial velocity along body z-axis
        self.phi0   = 0;  # initial roll angle
        self.theta0 = 0;  # initial pitch angle
        self.psi0   = 0;  # initial yaw angle
        self.p0     = 0;  # initial body frame roll rate
        self.q0     = 0;  # initial body frame pitch rate
        self.r0     = 0;  # initial body frame yaw rate
        # initial conditions for inverted pendulum
        self.pen_l    = 0.1; # m, the length of inverted pendulum
        self.pen_x    = 0; # initial displacement along iv in vehicle frame
        self.pen_y    = 0; # initial displacement along jv in vehicle frame
        self.pen_vx   = 0; # initial velocity along iv in vehicle frame
        self.pen_vy   = 0; # initial velocity along jv in vehicle frame

        # initial state
        self.x = 




























