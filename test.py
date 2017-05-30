#!/usr/bin/python2.7

from numpy import *
from param import Parameter

class uav_dynamics:
    P = Parameter()
    pn     = P.pn0
    pe     = P.pe0
    pd     = P.pd0
    u      = P.u0
    v      = P.v0
    w      = P.w0 
    phi    = P.phi0
    theta  = P.theta0
    psi    = P.psi0
    p      = P.p0
    q      = P.q0
    r      = P.r0
    pen_x  = P.pen_x0
    pen_y  = P.pen_y0
    pen_vx = P.pen_vx0
    pen_vy = P.pen_vy0
    Ts     = P.Ts
    Time   = 0.0

    def _init_(self):
        pass

    def step(self, uu):
     # input
        fx    = uu[0,0]
        fy    = uu[0,1]
        fz    = uu[0,2]
        taup  = uu[0,3] #tau phi
        taut  = uu[0,4] #tau theta
        taus  = uu[0,5] #tau psi
    
        sp = sin(self.phi)
        cp = cos(self.phi)
        st = sin(self.theta)
        ct = cos(self.theta)
        ss = sin(self.psi)
        cs = cos(self.psi)
        tt = tan(self.theta)

     # translational kinematics
        rotation_position = mat([[ct*cs, sp*st*cs-cp*ss, cp*st*cs+sp*ss], 
                                 [ct*ss, sp*st*ss+cp*cs, cp*st*ss-sp*cs],
                                 [-st,   sp*ct,          cp*ct]])
                                   
        position_dot = rotation_position * mat([self.u,self.v,self.w]).T
        pndot = position_dot[0,0]
        pedot = position_dot[1,0]
        pddot = position_dot[2,0]
    
     # translational dynamics
        udot = self.r*self.v - self.q*self.w + fx/self.P.mass
        vdot = self.p*self.w - self.r*self.u + fy/self.P.mass
        wdot = self.q*self.u - self.p*self.v + fz/self.P.mass
    
     # rotational kinematics
        rotation_angle = mat([[1, sp*tt, cp*tt],
                              [0, cp,    -sp],
                              [0, sp/ct, cp/ct]])
                      
        angle_dot = rotation_angle * mat([self.p, self.q, self.r]).T
        phidot   = angle_dot[0,0]
        thetadot = angle_dot[1,0]
        psidot   = angle_dot[2,0]

     # rorational dynamics
        pdot = (self.q*self.r*(self.P.Jy-self.P.Jz)/self.P.Jx) + (taup/self.P.Jx)
        qdot = (self.p*self.r*(self.P.Jz-self.P.Jx)/self.P.Jy) + (taut/self.P.Jy)
        rdot = (self.p*self.q*(self.P.Jx-self.P.Jy)/self.P.Jz) + (taus/self.P.Jz)

     # inverted pendulum kinematics
        pen_accel = rotation_position * mat([udot,vdot,wdot]).T
        xddot = pen_accel[0,0];
        yddot = pen_accel[1,0];
        zddot = pen_accel[2,0];

     # inverted pendulum dynamics
        pen_zeta = sqrt(self.P.pen_l**2.0 - self.pen_x**2.0 - self.pen_y**2.0)
        pen_xdot = self.pen_vx
        pen_ydot = self.pen_vy
        pen_alpha = (-pen_zeta**2.0/(pen_zeta**2.0+self.pen_x**2.0)) * (xddot+(pen_xdot**2.0*self.pen_x+pen_ydot**2.0*self.pen_x)/(pen_zeta**2.0) \
                  + (pen_xdot**2.0*self.pen_x**3.0+2*pen_xdot*pen_ydot*self.pen_x**2.0*self.pen_y+pen_ydot**2.0*self.pen_y**2.0*self.pen_x)/(pen_zeta**4.0) \
                  - (self.pen_x*(zddot+self.P.gravity))/(pen_zeta))
        pen_beta  = (-pen_zeta**2.0/(pen_zeta**2.0+self.pen_y**2.0)) * (yddot+(pen_ydot**2.0*self.pen_y+pen_xdot**2.0*self.pen_y)/(pen_zeta**2.0) \
                  + (pen_ydot**2.0*self.pen_y**3.0+2*pen_ydot*pen_xdot*self.pen_y**2.0*self.pen_x+pen_xdot**2.0*self.pen_x**2.0*self.pen_y)/(pen_zeta**4.0) \
                  - (self.pen_y*(zddot+self.P.gravity))/(pen_zeta))
        pen_vxdot = (pen_alpha - pen_beta*self.pen_x*self.pen_y/((self.P.pen_l**2.0-self.pen_y**2.0)*pen_zeta**2.0)) \
                  * (1 - (self.pen_x**2.0*self.pen_y**2.0)/((self.P.pen_l**2.0-self.pen_y**2.0)**2.0*pen_zeta**4.0))
        pen_vydot = pen_beta - (pen_vxdot*self.pen_x*self.pen_y)/(self.P.pen_l**2.0-self.pen_x**2.0)

        print "%.9f" %pndot
     # Update the quadcopter states
        self.pn    += pndot*self.P.Ts 
        self.pe    += pedot*self.P.Ts 
        self.pd    += pddot*self.P.Ts 
        self.u     += udot*self.P.Ts 
        self.v     += vdot*self.P.Ts 
        self.w     += wdot*self.P.Ts 
        self.phi   += phidot*self.P.Ts 
        self.theta += thetadot*self.P.Ts 
        self.psi   += psidot*self.P.Ts 
        self.p     += pdot*self.P.Ts 
        self.q     += qdot*self.P.Ts 
        self.r     += rdot*self.P.Ts 
     # Update the inverted pendulum states
        self.pen_x  += pen_xdot*self.P.Ts
        self.pen_y  += pen_ydot*self.P.Ts
        self.pen_vx += pen_vxdot*self.P.Ts
        self.pen_vy += pen_vydot*self.P.Ts
        self.Time   += self.P.Ts

    def reset(self):
        self.pn     = self.P.pn0
        self.pe     = self.P.pe0
        self.pd     = self.P.pd0
        self.u      = self.P.u0
        self.v      = self.P.v0
        self.w      = self.P.w0 
        self.phi    = self.P.phi0
        self.theta  = self.P.theta0
        self.psi    = self.P.psi0
        self.p      = self.P.p0
        self.q      = self.P.q0
        self.r      = self.P.r0
        self.pen_x  = self.P.pen_x0
        self.pen_y  = self.P.pen_y0
        self.pen_vx = self.P.pen_vx0
        self.pen_vy = self.P.pen_vy0
        self.Ts     = self.P.Ts
        self.Time   = 0.0  
# end class