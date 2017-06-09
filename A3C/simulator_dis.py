#!/usr/bin/python2.7
# Filename: simulator.py
# Description: This is the simulator of quadcopter with an inverted pendulum on it
# Auther: Peng Wei, Zhenghao Fei


import numpy as np
from scipy.integrate import odeint

class QuadCopter(object):
    def __init__(self, Ts=0.01, max_time = 10, inverted_pendulum=True):
    # simulator  step time
        self.Ts          = Ts
        self.max_time = max_time
        self.stateSpace  = 22
        self.actionSpace = 6
        self.actionLimit  = 5.0 # maximum rotor speed degree/s TBD
        self.inverted_pendulum = inverted_pendulum

    # physical parameters of airframe
        # self.gravity = 9.81
        # self.l       = 0.175  # m, Distance between rotor and center
        # self.pen_l   = 0.20  # m, the length of stick
        # self.k1      = 1.0      # propellers constant
        # self.k2      = 2.0      # propellers constant 
        # self.mass    = 0.5  # mass
        # self.Jx      = 2.32e-3
        # self.Jy      = 2.32e-3
        # self.Jz      = 4.00e-3

        self.gravity = 9.81
        self.l = 0.2; # m, Distance between rotor and center
        self.pen_l   = 0.20  # m, the length of stick
        self.k1 = 100; # propellers constant
        self.k2 = 100; # propellers constant 
        self.R = 0.04; # m, Center mass radius 
        self.M = 1 # kg, Body weight
        self.m = 0.07 #kg, Rotor weight 
        self.mass = self.M + self.m;
        self.Jx   = 2*self.M*self.R**2/5 + 2*self.l*self.m;
        self.Jy   = 2*self.M*self.R**2/5 + 2*self.l*self.m;
        self.Jz   = 2*self.M*self.R**2/5 + 4*self.l*self.m;

    # initial conditions
        self.pn0    = 0.0  # initial North position
        self.pe0    = 0.0  # initial East position
        self.pd0    = 0.0  # initial Down position (negative altitude)
        self.u0     = 0.0  # initial velocity along body x-axis
        self.v0     = 0.0  # initial velocity along body y-axis
        self.w0     = 0.0  # initial velocity along body z-axis
        self.phi0   = 0.0  # initial roll angle
        self.theta0 = 0.0  # initial pitch angle
        self.psi0   = 0.0  # initial yaw angle
        self.p0     = 0.0  # initial body frame roll rate
        self.q0     = 0.0  # initial body frame pitch rate
        self.r0     = 0.0  # initial body frame yaw rate

    # initial conditions for inverted pendulum
        self.pen_x0    = 0.0 # initial displacement along iv in vehicle frame
        self.pen_y0    = 0.0 # initial displacement along jv in vehicle frame
        self.pen_vx0   = 0.0 # initial velocity along iv in vehicle frame
        self.pen_vy0   = 0.0 # initial velocity along jv in vehicle frame

    # maximum conditions
        self.pn_max    =  100  # max North position
        self.pe_max    =  100  # max East position
        self.pd_max    =  100  # max Down position (negative altitude)
        self.u_max     = 10 # max velocity along body x-axis
        self.v_max     = 10 # max velocity along body y-axis
        self.w_max     = 10 # max velocity along body z-axis
        self.p_max     = 10 # max body frame roll rate
        self.q_max     = 10 # max body frame pitch rate
        self.r_max     = 10 # max body frame yaw rate
        

    # apply initial conditions
        self.reset()

    def reset(self):
        # print "system reset"
        self.pn     = self.pn0
        self.pe     = self.pe0
        self.pd     = self.pd0
        self.u      = self.u0
        self.v      = self.v0
        self.w      = self.w0 
        self.phi    = self.phi0
        self.theta  = self.theta0
        self.psi    = self.psi0
        self.p      = self.p0
        self.q      = self.q0
        self.r      = self.r0
        self.pen_x  = self.pen_x0
        self.pen_y  = self.pen_y0
        self.pen_vx = self.pen_vx0
        self.pen_vy = self.pen_vy0
        self.time   = 0.0 # simulation time
        self.uu = np.zeros(6, dtype=np.float32)

        self.states = np.asarray([self.pn, self.pe, self.pd, self.u, self.v, self.w, self.phi, self.theta, self.psi,
                                  self.p,  self.q,  self.r,  self.pen_x,  self.pen_y,  self.pen_vx,  self.pen_vy])
        states_out = np.concatenate((self.states, self.uu))
        return states_out

    def force(self, x):
        f = self.k1 * x
        return f 

    def torque(self, x):
        tau = self.k2 * x
        return tau

    def trunc_error(self,x):
        if np.absolute(x) < 1e-15:
            return 0.0
        else:
            return x

    def forces_moments(self, delta_f, delta_r, delta_b, delta_l, theta, phi):
        Force_x = -self.mass * self.gravity * np.sin(theta);
        Force_y =  self.mass * self.gravity * np.cos(theta) * np.sin(phi)
        Force_z =  self.mass * self.gravity * np.cos(theta) * np.cos(phi) \
                - (self.force(delta_f)+self.force(delta_r)+self.force(delta_b)+self.force(delta_l))

        Torque_x = -self.l * self.force(delta_r) + self.l * self.force(delta_l)     
        Torque_y = self.l * self.force(delta_f) - self.l * self.force(delta_b)
        Torque_z = -self.torque(delta_f) + self.torque(delta_r) - self.torque(delta_b) + self.torque(delta_l)

        uu = np.asarray([self.trunc_error(Force_x), self.trunc_error(Force_y), self.trunc_error(Force_z),
                         self.trunc_error(Torque_x), self.trunc_error(Torque_y), self.trunc_error(Torque_z)])

        return uu

    def action_trans(self, a):
        delta = 0.5
        if a == 0:
            self.uu[0] += delta;

        if a == 1:
            self.uu[0] -= delta;

        if a == 2:
            self.uu[1] += delta;

        if a == 3:
            self.uu[1] -= delta; 

        if a == 4:
            self.uu[2] += delta;

        if a == 5:
            self.uu[2] -= delta;  

        # action limit
        self.uu = np.clip(self.uu, -self.actionLimit, self.actionLimit)

        return self.uu

    def Derivative(self, states, t, uu):
    # state variables
        pn     = states[0]    
        pe     = states[1]    
        pd     = states[2]   
        u      = states[3]     
        v      = states[4]    
        w      = states[5]   
        phi    = states[6] 
        theta  = states[7] 
        psi    = states[8]  
        p      = states[9]    
        q      = states[10]   
        r      = states[11]    
        pen_x  = states[12] 
        pen_y  = states[13] 
        pen_vx = states[14] 
        pen_vy = states[15] 
    # control inputs
        # uu    = self.forces_moments(delta_f, delta_r, delta_b, delta_l, theta, phi)
        fx    = uu[0]
        fy    = uu[1]
        fz    = uu[2]
        taup  = uu[3] #tau phi
        taut  = uu[4] #tau theta
        taus  = uu[5] #tau psi
        
        # print fx, fy, fz

        sp = np.sin(phi)
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)
        ss = np.sin(psi)
        cs = np.cos(psi)
        tt = np.tan(theta)

     # translational kinematics
        rotation_position = np.mat([[ct*cs, sp*st*cs-cp*ss, cp*st*cs+sp*ss], 
                                    [ct*ss, sp*st*ss+cp*cs, cp*st*ss-sp*cs],
                                    [-st,   sp*ct,          cp*ct]])
                                   
        position_dot = rotation_position * np.mat([u,v,w]).T
        pndot = position_dot[0,0]
        pedot = position_dot[1,0]
        pddot = position_dot[2,0]
    
     # translational dynamics
        udot = r*v - q*w + fx/self.mass
        vdot = p*w - r*u + fy/self.mass
        wdot = q*u - p*v + fz/self.mass
    
     # rotational kinematics
        rotation_angle = np.mat([[1, sp*tt, cp*tt],
                                 [0, cp,    -sp],
                                 [0, sp/ct, cp/ct]])
                      
        angle_dot = rotation_angle * np.mat([p, q, r]).T
        phidot    = angle_dot[0,0]
        thetadot  = angle_dot[1,0]
        psidot    = angle_dot[2,0]

     # rotational dynamics
        pdot = (q*r*(self.Jy-self.Jz)/self.Jx) + (taup/self.Jx)
        qdot = (p*r*(self.Jz-self.Jx)/self.Jy) + (taut/self.Jy)
        rdot = (p*q*(self.Jx-self.Jy)/self.Jz) + (taus/self.Jz)

     # inverted pendulum kinematics
        pen_accel = rotation_position * np.mat([udot,vdot,wdot]).T
        xddot     = pen_accel[0,0]
        yddot     = pen_accel[1,0]
        zddot     = pen_accel[2,0]

     # inverted pendulum dynamics
        if self.inverted_pendulum: 
            pen_zeta  = np.sqrt(self.pen_l**2.0 - pen_x**2.0 - pen_y**2.0)
            if pen_zeta <=0:
                self.terminated = True
                self.info = "pen_zeta<0"
            pen_xdot  = pen_vx
            pen_ydot  = pen_vy
            pen_alpha = (-pen_zeta**2.0/(pen_zeta**2.0+pen_x**2.0)) * (xddot+(pen_xdot**2.0*pen_x+pen_ydot**2.0*pen_x)/(pen_zeta**2.0) \
                      + (pen_xdot**2.0*pen_x**3.0+2*pen_xdot*pen_ydot*pen_x**2.0*pen_y+pen_ydot**2.0*pen_y**2.0*pen_x)/(pen_zeta**4.0) \
                      - (pen_x*(zddot+self.gravity))/(pen_zeta))
            pen_beta  = (-pen_zeta**2.0/(pen_zeta**2.0+pen_y**2.0)) * (yddot+(pen_ydot**2.0*pen_y+pen_xdot**2.0*pen_y)/(pen_zeta**2.0) \
                      + (pen_ydot**2.0*pen_y**3.0+2*pen_ydot*pen_xdot*pen_y**2.0*pen_x+pen_xdot**2.0*pen_x**2.0*pen_y)/(pen_zeta**4.0) \
                      - (pen_y*(zddot+self.gravity))/(pen_zeta))
            pen_vxdot = (pen_alpha - pen_beta*pen_x*pen_y/((self.pen_l**2.0-pen_y**2.0)*pen_zeta**2.0)) \
                      * (1.0 - (pen_x**2.0*pen_y**2.0)/((self.pen_l**2.0-pen_y**2.0)**2.0*pen_zeta**4.0))
            pen_vydot = pen_beta - (pen_vxdot*pen_x*pen_y)/((self.pen_l**2.0-pen_x**2.0)*pen_zeta**2.0)
        else:
            pen_zeta  = 0
            pen_xdot  = 0
            pen_ydot  = 0
            pen_alpha = 0
            pen_beta  = 0
            pen_vxdot = 0
            pen_vydot = 0

        states_dot = np.asarray([pndot, pedot,  pddot,  udot,   vdot,   wdot,   phidot, thetadot,   psidot, pdot,   qdot,   rdot,
                                 pen_xdot,  pen_ydot,   pen_vxdot,  pen_vydot])
        return states_dot

    def naive_int(self, derivative_func, states, Ts, uu):

        states_dot = derivative_func(states, Ts, uu)
        states += states_dot*Ts
        sol =  np.vstack((states_dot, states))
        return sol


    def step(self, action):
        self.terminated = False
        self.info = 'normal'

        # procecss action to uu
        self.action_trans(action)
        # delta   = np.asarray(delta)*3.1416/180
        # delta_f = delta[0]
        # delta_r = delta[1]
        # delta_b = delta[2]
        # delta_l = delta[3]

    # integral, ode
        # sol = odeint(self.Derivative, self.states, [self.time, self.time+self.Ts], args=(delta_f,delta_r,delta_b,delta_l), full_output=False, printmessg=False)
        sol = self.naive_int(self.Derivative, self.states, self.Ts, self.uu)

        self.pn     = sol[1,0] 
        self.pe     = sol[1,1] 
        self.pd     = sol[1,2] 
        self.u      = sol[1,3]
        self.v      = sol[1,4]   
        self.w      = sol[1,5]
        self.phi    = sol[1,6]
        self.theta  = sol[1,7]
        self.psi    = sol[1,8]
        self.p      = sol[1,9]  
        self.q      = sol[1,10]  
        self.r      = sol[1,11]  
        self.pen_x  = sol[1,12]
        self.pen_y  = sol[1,13]
        self.pen_vx = sol[1,14]
        self.pen_vy = sol[1,15]
        self.time   += self.Ts

        # check boundry condition
        if self.pn > self.pn_max:
            self.pn = self.pn_max
        if self.pn < -self.pn_max:
            self.pn = -self.pn_max

        if self.pe > self.pe_max:
            self.pe = self.pe_max
        if self.pe < -self.pe_max:
            self.pe = -self.pe_max

        if self.pd > self.pd_max:
            self.pd = self.pd_max
        if self.pd < -self.pd_max:
            self.pd = -self.pd_max

        if self.u > self.u_max:
            self.u = self.u_max
        if self.u < -self.u_max:
            self.u = -self.u_max

        if self.v > self.v_max:
            self.v = self.v_max
        if self.v < -self.v_max:
            self.v = -self.v_max

        if self.w > self.w_max:
            self.w = self.w_max
        if self.w < -self.w_max:
            self.w = -self.w_max

        if self.p > self.p_max:
            self.p = self.p_max
        if self.p < -self.p_max:
            self.p = -self.p_max

        if self.q > self.q_max:
            self.q = self.q_max
        if self.q < -self.q_max:
            self.q = -self.q_max

        if self.r > self.r_max:
            self.r = self.r_max
        if self.r < -self.r_max:
            self.r = -self.r_max

        if self.time > self.max_time:
            self.terminated = True
            self.info = 'timeout'   
    # # Fail condition check
    #     if self.pd > 0:
    #         self.terminated = True
    #         self.info = 'crash'   

        # print 'Time = %f' %self.time
        self.states = np.asarray([self.pn, self.pe, self.pd, self.u, self.v, self.w, self.phi, self.theta, self.psi,
                                  self.p,  self.q,  self.r,  self.pen_x,  self.pen_y,  self.pen_vx,  self.pen_vy])
        states_out = np.concatenate((self.states, self.uu))

        if  self.terminated:
            self.reset()

        return states_out, self.terminated, self.info

# end class