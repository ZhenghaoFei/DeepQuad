
function drawAircraft(uu,V,F,patchcolors)

    % process inputs to function
    pn       = uu(1);       % inertial North position     
    pe       = uu(2);       % inertial East position
    pd       = uu(3);           
    u        = uu(4);       
    v        = uu(5);       
    w        = uu(6);       
    phi      = uu(7);       % roll angle         
    theta    = uu(8);       % pitch angle     
    psi      = uu(9);       % yaw angle     
    p        = uu(10);       % roll rate
    q        = uu(11);       % pitch rate     
    r        = uu(12);       % yaw rate  
    % inverted pendulum
    pen_x    = uu(13);
    pen_y    = uu(14);
    pen_vx   = uu(15);
    pen_vy   = uu(15); 

    t        = uu(17);       % time
    
    % define persistent variables 
    persistent vehicle_handle;
    persistent Vertices
    persistent Faces
    persistent facecolors
    
    % first time function is called, initialize plot and persistent vars
    if t==0,
        figure(1), clf
        [Vertices,Faces,facecolors] = defineVehicleBody;
        vehicle_handle = drawVehicleBody(Vertices,Faces,facecolors,...
                                               pn,pe,pd,phi,theta,psi,...
                                               [],'normal');
        title('Vehicle')
        xlabel('East')
        ylabel('North')
        zlabel('-Down')
        view(32,47)  % set the vieew angle for figure
        axis_size = 30;
        axis([-axis_size,axis_size,-axis_size,axis_size,-axis_size,axis_size]);
        grid on;
        hold on
        
    % at every other time step, redraw base and rod
    else 
        drawVehicleBody(Vertices,Faces,facecolors,...
                           pn,pe,pd,phi,theta,psi,...
                           vehicle_handle);
    end
end

  
%=======================================================================
% drawVehicle
% return handle if 3rd argument is empty, otherwise use 3rd arg as handle
%=======================================================================
%
function handle = drawVehicleBody(V,F,patchcolors,...
                                     pn,pe,pd,phi,theta,psi,...
                                     handle,mode)
  V = rotate(V, phi, theta, psi);  % rotate vehicle
  V = translate(V, pn, pe, pd);  % translate vehicle
  % transform vertices from NED to XYZ (for matlab rendering)
  R = [...
      0, 1, 0;...
      1, 0, 0;...
      0, 0, -1;...
      ];
  V = R*V;
  
  if isempty(handle),
  handle = patch('Vertices', V', 'Faces', F,...
                 'FaceVertexCData',patchcolors,...
                 'FaceColor','flat',...
                 'EraseMode', mode);
  else
    set(handle,'Vertices',V','Faces',F);
    drawnow
  end
end

%%%%%%%%%%%%%%%%%%%%%%%
function pts=rotate(pts,phi,theta,psi)

  % define rotation matrix (right handed)
  R_roll = [...
          1, 0, 0;...
          0, cos(phi), sin(phi);...
          0, -sin(phi), cos(phi)];
  R_pitch = [...
          cos(theta), 0, -sin(theta);...
          0, 1, 0;...
          sin(theta), 0, cos(theta)];
  R_yaw = [...
          cos(psi), sin(psi), 0;...
          -sin(psi), cos(psi), 0;...
          0, 0, 1];
  R = R_roll*R_pitch*R_yaw;  
    % note that R above either leaves the vector alone or rotates
    % a vector in a left handed rotation.  We want to rotate all
    % points in a right handed rotation, so we must transpose
  R = R';

  % rotate vertices
  pts = R*pts;
  
end
% end rotateVert

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% translate vertices by pn, pe, pd
function pts = translate(pts,pn,pe,pd)

  pts = pts + repmat([pn;pe;pd],1,size(pts,2));
  
end

% end translate


%=======================================================================
% defineVehicleBody
%=======================================================================
function [V,F,facecolors] = defineVehicleBody

% Define the body vertices of the vehicle
size_b = 0.4;
B  = [...
      size_b, size_b, 0;... % B 1
      -size_b, size_b, 0;... % B 2
      -size_b, -size_b, 0;... % B 3
      size_b, -size_b, 0;... % B 4
];

% Define the rotor vertices of the vehicle
size_c = 0.04;
size_cop = 2*size_b;
C  = [...
      size_cop, 0, 0;... % C 1
      size_c, size_c, 0;... % C 5
      0, size_cop, 0;... % C 2
      -size_c, size_c, 0;... % C 6
      -size_cop, 0, 0;... % C 3
      -size_c, -size_c, 0;... % C 7
      0, -size_cop, 0;... % C 4
      size_c, -size_c, 0;... % C 8
];
R_cop = [...
          cos([pi/4]), sin(pi/4), 0;...
          -sin(pi/4), cos(pi/4), 0;...
          0, 0, 1];

l = 2; % length from body to copters

% Vertices of 4 copters
C1 = C*R_cop + [l, 0, 0];
C2 = C*R_cop + [0, l ,0];
C3 = C*R_cop + [-l, 0, 0];
C4 = C*R_cop + [0, -l, 0];

% Define the connection vertices of the vehicle
w_co = 0.01; % width of connectionsc
h_co = 0;
Co1 = [...
              l, w_co, h_co; ...
              h_co, w_co, h_co; ...
              h_co, -w_co, h_co; ...
              l, -w_co, h_co; ...
              ];
Co2 = [...
              w_co, l, h_co; ...
              -w_co, l, h_co; ...
              -w_co, h_co, h_co; ...
              w_co, h_co, h_co; ...
              ];

Co3 = [...
              -l, w_co, h_co; ...
              h_co, w_co, h_co; ...
              h_co, -w_co, h_co; ...
              -l, -w_co, h_co; ...
              ];
Co4 = [...
              w_co, -l, h_co; ...
              -w_co, -l, h_co; ...
              -w_co, h_co, h_co; ...
              w_co, h_co, h_co; ...
              ];

% inverted pendulum
% pen_l = 1;
% pen = []

% Define the vertices (physical location of vertices
V = [...
    B; ... %pt 1-4
    C1; ... %pt 5-12
    C2; ... %pt 13-20
    C3; ... %pt 21-28
    C4; ... %pt 28-36
    Co1; ... %pt 37-40
    Co2; ... %pt 41-44
    Co3; ... %pt 45-48
    Co4; ... %pt 49-52
    ]'; 

R_V = [...
          cos([0]), sin(0), 0;...
          -sin(0), cos(0), 0;...
          0, 0, 1];

V = R_V * V; 
% define faces as a list of vertices numbered above
F = [...
        1, 1, 2, 2, 3, 3, 4, 4;...  % body
        5, 6, 7, 8, 9, 10, 11, 12; ... % copter 1
        13, 14, 15, 16, 17, 18, 19, 20; ... % copter 2
        21, 22, 23, 24, 25, 26, 27, 28; ... % copter 3
        29, 30, 31, 32, 33, 34, 35, 36; ... % copter 4
        37, 37, 38, 38, 39, 39, 40, 40; ... % connection 1
        41, 41, 42, 42, 43, 43, 44, 44; ... % connection 2
        45, 45, 46, 46, 47, 47, 48, 48; ... % connection 3
        49, 49, 50, 50, 51, 51, 52, 52; ... % connection 4
        ];

% define colors for each face
  myblk = [0, 0, 0];    
  mywhite = [1, 1, 1];    
  myred = [1, 0, 0];
  mygreen = [0, 1, 0];
  myblue = [0, 0, 1];
  myyellow = [1, 1, 0];
  mycyan = [0, 1, 1];

  facecolors = [...
        mygreen;...  % body
        myblk; ... % copter 1
        myblk; ... % copter 2
        myblk; ... % copter 3
        myblk; ... % copter 4
        mywhite; ... % connection 1
        mywhite; ... % connection 2
        mywhite; ... % connection 3
        mywhite; ... % connection 4
    ];
end
  