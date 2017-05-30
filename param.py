#!/usr/bin/python2.7

class Parameter:
	gravity = 9.81;
	Ts = 0.01;

	#physical parameters of airframe
	l 	 = 0.2; # m, Distance between rotor and center
	k1   = 100.0; # propellers constant
	k2   = 100.0; # propellers constant 
	R 	 = 0.04; # m, Center mass radius 
	M 	 = 1.0 # kg, Body weight
	m 	 = 0.07 #kg, Rotor weight 
	mass = M + m;
	Jx   = 2.0*M*R**2.0/5 + 2.0*l*m;
	Jy   = 2.0*M*R**2.0/5 + 2.0*l*m;
	Jz   = 2.0*M*R**2.0/5 + 4.0*l*m;
	Jxz  = 0.0;

	# initial conditions
	pn0    = 0.0;  # initial North position
	pe0    = 0.0;  # initial East position
	pd0    = 0.0;  # initial Down position (negative altitude)
	u0     = 0.0;  # initial velocity along body x-axis
	v0     = 0.0;  # initial velocity along body y-axis
	w0     = 0.0;  # initial velocity along body z-axis
	phi0   = 0.0;  # initial roll angle
	theta0 = 0.0;  # initial pitch angle
	psi0   = 0.0;  # initial yaw angle
	p0     = 0.0;  # initial body frame roll rate
	q0     = 0.0;  # initial body frame pitch rate
	r0     = 0.0;  # initial body frame yaw rate

	# initial conditions for inverted pendulum
	pen_l     = 0.1; # m, the length of inverted pendulum
	pen_x0    = 0.0; # initial displacement along iv in vehicle frame
	pen_y0    = 0.0; # initial displacement along jv in vehicle frame
	pen_vx0   = 0.0; # initial velocity along iv in vehicle frame
	pen_vy0   = 0.0; # initial velocity along jv in vehicle frame

	def _init_(self):
		pass