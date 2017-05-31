clear; close all; clc;
P.gravity = 9.81;
   
%physical parameters of airframe
P.l = 0.2; % m, Distance between rotor and center
P.R = 0.04; % m, Center mass radius 
P.M = 1 % kg, Body weight
P.m = 0.07 %kg, Rotor weight 
P.mass = P.M + P.m;
P.Jx   = 2*P.M*P.R^2/5 + 2*P.l*P.m;
P.Jy   = 2*P.M*P.R^2/5 + 2*P.l*P.m;
P.Jz   = 2*P.M*P.R^2/5 + 4*P.l*P.m;
P.Jxz  = 0;

% P.r = P.Jx*P.Jz-P.Jxz^2;
% P.r1 = P.Jxz*(P.Jx-P.Jy+P.Jz)/P.r;
% P.r2 = P.Jz*(P.Jz-P.Jy)+P.Jxz^2;
% P.r3 = P.Jz/P.r;
% P.r4 = P.Jxz/P.r;
% P.r5 = (P.Jz-P.Jx)/P.Jy;
% P.r6 = P.Jxz/P.Jy;
% P.r7 = ((P.Jx-P.Jy)*P.Jx+P.Jxz^2)/P.r;
% P.r8 = P.Jx/P.r;


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

