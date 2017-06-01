#!/usr/bin/python2.7
# Filename: simulator.py
# Description: This is the simulator of quadcopter with an inverted pendulum on it
# Auther: Peng Wei, Zhenghao Fei


import numpy as np

class QuadCopter(object):
    def __init__(self, Ts=0.01):
        # simulator property
        self.Ts          = Ts
        self.stateSpace  = 16
        self.actionSpace = 4
        self.actionLimit  = 10 # maximum rotor speed degree/s TBD

        # physical parameters of airframe
        self.gravity = 9.81
        self.l       = 0.2      # m, Distance between rotor and center
        self.k1      = 100.0    # propellers constant
        self.k2      = 100.0    # propellers constant 
        self.R       = 0.04     # m, Center mass radius 
        self.M       = 1.0      # kg, Body weight
        self.m       = 0.07     # kg, Rotor weight 
        self.mass    = self.M + self.m
        self.Jx      = 2.0*self.M*self.R**2.0/5 + 2.0*self.l*self.m
        self.Jy      = 2.0*self.M*self.R**2.0/5 + 2.0*self.l*self.m
        self.Jz      = 2.0*self.M*self.R**2.0/5 + 4.0*self.l*self.m
        self.Jxz     = 0.0

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
        self.pen_l     = 0.1 # m, the length of inverted pendulum
        self.pen_x0    = 0.0 # initial displacement along iv in vehicle frame
        self.pen_y0    = 0.0 # initial displacement along jv in vehicle frame
        self.pen_vx0   = 0.0 # initial velocity along iv in vehicle frame
        self.pen_vy0   = 0.0 # initial velocity along jv in vehicle frame

        # apply initial conditions
        self.reset()

    def reset(self):
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
        self.time   = 0.0  

        states = np.asarray([
        self.pn,    
        self.pe,    
        self.pd,    
        self.u,     
        self.v,     
        self.w,     
        self.phi,   
        self.theta, 
        self.psi,   
        self.p,     
        self.q,     
        self.r,     
        self.pen_x, 
        self.pen_y, 
        self.pen_vx,
        self.pen_vy,]
        )
        return states

    def forces_moments(self, delta):
        delta = np.asarray(delta)*3.142/180
        delta = delta.reshape([4,1])
        # Mapping from propellers to uu
        sp = np.sin(self.phi)
        cp = np.cos(self.phi)
        st = np.sin(self.theta)
        ct = np.cos(self.theta)
        ss = np.sin(self.psi)
        cs = np.cos(self.psi)
        tt = np.tan(self.theta)

        #  M matrix
        M = np.mat([[self.k1, self.k1, self.k1, self.k1],
             [0, -self.l*self.k1, 0, self.l*self.k1],
             [self.l*self.k1, 0, -self.l*self.k1, 0],
             [-self.k2, self.k2, -self.k2, self.k2]
             ])
        
        #  compute external forces and torques on aircraft
        F_T = M*delta;

        F = F_T[0];
        Torque = F_T[1:4]

        f_gravity = np.mat([[-self.gravity*self.mass*st],
                     [self.gravity*self.mass*ct*sp],
                     [self.gravity*self.mass*ct*cp]
                     ])
        Force = f_gravity - np.asarray([[0], [0] , [F]]);
        uu = np.vstack((Force, Torque))
        return uu

    def step(self, delta):
        terminated = False
        info = 'normal'
     # input
        uu = self.forces_moments(delta)
        fx    = uu[0, 0]
        fy    = uu[1, 0]
        fz    = uu[2, 0]
        taup  = uu[3, 0] #tau phi
        taut  = uu[4, 0] #tau theta
        taus  = uu[5, 0] #tau psi
    
        sp = np.sin(self.phi)
        cp = np.cos(self.phi)
        st = np.sin(self.theta)
        ct = np.cos(self.theta)
        ss = np.sin(self.psi)
        cs = np.cos(self.psi)
        tt = np.tan(self.theta)

     # translational kinematics
        rotation_position = np.mat([[ct*cs, sp*st*cs-cp*ss, cp*st*cs+sp*ss], 
                                    [ct*ss, sp*st*ss+cp*cs, cp*st*ss-sp*cs],
                                    [-st,   sp*ct,          cp*ct]])
                                   
        position_dot = rotation_position * np.mat([self.u,self.v,self.w]).T
        pndot = position_dot[0,0]
        pedot = position_dot[1,0]
        pddot = position_dot[2,0]
    
     # translational dynamics
        udot = self.r*self.v - self.q*self.w + fx/self.mass
        vdot = self.p*self.w - self.r*self.u + fy/self.mass
        wdot = self.q*self.u - self.p*self.v + fz/self.mass
    
     # rotational kinematics
        rotation_angle = np.mat([[1, sp*tt, cp*tt],
                                 [0, cp,    -sp],
                                 [0, sp/ct, cp/ct]])
                      
        angle_dot = rotation_angle * np.mat([self.p, self.q, self.r]).T
        phidot    = angle_dot[0,0]
        thetadot  = angle_dot[1,0]
        psidot    = angle_dot[2,0]

     # rorational dynamics
        pdot = (self.q*self.r*(self.Jy-self.Jz)/self.Jx) + (taup/self.Jx)
        qdot = (self.p*self.r*(self.Jz-self.Jx)/self.Jy) + (taut/self.Jy)
        rdot = (self.p*self.q*(self.Jx-self.Jy)/self.Jz) + (taus/self.Jz)

     # inverted pendulum kinematics
        pen_accel = rotation_position * np.mat([udot,vdot,wdot]).T
        xddot     = pen_accel[0,0]
        yddot     = pen_accel[1,0]
        zddot     = pen_accel[2,0]

     # inverted pendulum dynamics
        # pen_zeta  = np.sqrt(self.pen_l**2.0 - self.pen_x**2.0 - self.pen_y**2.0)
        # pen_xdot  = self.pen_vx
        # pen_ydot  = self.pen_vy
        # pen_alpha = (-pen_zeta**2.0/(pen_zeta**2.0+self.pen_x**2.0)) * (xddot+(pen_xdot**2.0*self.pen_x+pen_ydot**2.0*self.pen_x)/(pen_zeta**2.0) \
        #           + (pen_xdot**2.0*self.pen_x**3.0+2*pen_xdot*pen_ydot*self.pen_x**2.0*self.pen_y+pen_ydot**2.0*self.pen_y**2.0*self.pen_x)/(pen_zeta**4.0) \
        #           - (self.pen_x*(zddot+self.gravity))/(pen_zeta))
        # pen_beta  = (-pen_zeta**2.0/(pen_zeta**2.0+self.pen_y**2.0)) * (yddot+(pen_ydot**2.0*self.pen_y+pen_xdot**2.0*self.pen_y)/(pen_zeta**2.0) \
        #           + (pen_ydot**2.0*self.pen_y**3.0+2*pen_ydot*pen_xdot*self.pen_y**2.0*self.pen_x+pen_xdot**2.0*self.pen_x**2.0*self.pen_y)/(pen_zeta**4.0) \
        #           - (self.pen_y*(zddot+self.gravity))/(pen_zeta))
        # pen_vxdot = (pen_alpha - pen_beta*self.pen_x*self.pen_y/((self.pen_l**2.0-self.pen_y**2.0)*pen_zeta**2.0)) \
        #           * (1 - (self.pen_x**2.0*self.pen_y**2.0)/((self.pen_l**2.0-self.pen_y**2.0)**2.0*pen_zeta**4.0))
        # pen_vydot = pen_beta - (pen_vxdot*self.pen_x*self.pen_y)/(self.pen_l**2.0-self.pen_x**2.0)
        pen_zeta  = 0
        pen_xdot  = 0
        pen_ydot  = 0
        pen_alpha = 0
        pen_beta  = 0
        pen_vxdot = 0
        pen_vydot = 0
     # Update the quadcopter states
        self.pn    += pndot*self.Ts 
        self.pe    += pedot*self.Ts 
        self.pd    += pddot*self.Ts 
        self.u     += udot*self.Ts 
        self.v     += vdot*self.Ts 
        self.w     += wdot*self.Ts 
        self.phi   += phidot*self.Ts 
        self.theta += thetadot*self.Ts 
        self.psi   += psidot*self.Ts 
        self.p     += pdot*self.Ts 
        self.q     += qdot*self.Ts 
        self.r     += rdot*self.Ts 
     # Update the inverted pendulum states
        self.pen_x  += pen_xdot*self.Ts
        self.pen_y  += pen_ydot*self.Ts
        self.pen_vx += pen_vxdot*self.Ts
        self.pen_vy += pen_vydot*self.Ts
        self.time   += self.Ts

        states = np.asarray([
        self.pn,    
        self.pe,    
        self.pd,    
        self.u,     
        self.v,     
        self.w,     
        self.phi,   
        self.theta, 
        self.psi,   
        self.p,     
        self.q,     
        self.r,     
        self.pen_x, 
        self.pen_y, 
        self.pen_vx,
        self.pen_vy,]
        )

        if  terminated:
            info = 'terminated'
            self.reset()

        return states, terminated, info












