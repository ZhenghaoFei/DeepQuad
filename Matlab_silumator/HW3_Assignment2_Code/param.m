clear; close all; clc;
P.gravity = 9.81;
P.Ts = 0.01;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%physical parameters of airframe
P.l = 0.2; % m, Distance between rotor and center
P.k1 = 100; % propellers constant
P.k2 = 100; % propellers constant 
P.R = 0.04; % m, Center mass radius 
P.M = 1 % kg, Body weight
P.m = 0.07 %kg, Rotor weight 
P.mass = P.M + P.m;
P.Jx   = 2*P.M*P.R^2/5 + 2*P.l*P.m;
P.Jy   = 2*P.M*P.R^2/5 + 2*P.l*P.m;
P.Jz   = 2*P.M*P.R^2/5 + 4*P.l*P.m;
P.Jxz  = 0;

% initial conditions
P.pn0    = 0;  % initial North position
P.pe0    = 0;  % initial East position
P.pd0    = 0;  % initial Down position (negative altitude)
P.u0     = 0;  % initial velocity along body x-axis
P.v0     = 0;  % initial velocity along body y-axis
P.w0     = 0;  % initial velocity along body z-axis
P.phi0   = 0;  % initial roll angle
P.theta0 = 0;  % initial pitch angle
P.psi0   = 0;  % initial yaw angle
P.p0     = 0;  % initial body frame roll rate
P.q0     = 0;  % initial body frame pitch rate
P.r0     = 0;  % initial body frame yaw rate

% initial conditions for inverted pendulum
P.pen_l    = 0.1; % m, the length of inverted pendulum
P.pen_x    = 0; % initial displacement along iv in vehicle frame
P.pen_y    = 0; % initial displacement along jv in vehicle frame
P.pen_vx   = 0; % initial velocity along iv in vehicle frame
P.pen_vy   = 0; % initial velocity along jv in vehicle frame




