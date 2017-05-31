% forces_moments.m
%   Computes the forces and moments acting on the airframe. 
%
%   Output is
%       F     - forces
%       M     - moments

function out = forces_moments(x, delta, P)

    % relabel the inputs
    pn      = x(1);
    pe      = x(2);
    pd      = x(3);
    u       = x(4);
    v       = x(5);
    w       = x(6);
    phi     = x(7);
    theta   = x(8);
    psi     = x(9);
    p       = x(10);
    q       = x(11);
    r       = x(12);
    delta_f = delta(1);
    delta_r = delta(2);
    delta_b = delta(3);
    delta_l = delta(4);

    % rotation matrix from vehicle to body
    sp = sin(phi);
    cp = cos(phi);
    st = sin(theta);
    ct = cos(theta);
    ss = sin(psi);
    cs = cos(psi);
    
    
    
    % M matrix
    M = [P.k1, P.k1, P.k1, P.k1;
         0, -P.l*P.k1, 0, P.l*P.k1;
         P.l*P.k1, 0, -P.l*P.k1, 0;
         -P.k2, P.k2, -P.k2, P.k2;
         ];
    
    % compute external forces and torques on aircraft
    delta = [delta_f;
             delta_r;
             delta_b;
             delta_l;
             ];

    F_T = M*delta;

    F = F_T(1);
    Torque = F_T(2:4);

    f_gravity = [-P.gravity*P.mass*st;
                 P.gravity*P.mass*ct*sp;
                 P.gravity*P.mass*ct*cp;
                 ];
    Force = f_gravity - [0; 0; F;];

    out = [Force; Torque];
end