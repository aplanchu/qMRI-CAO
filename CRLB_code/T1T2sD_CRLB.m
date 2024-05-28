% Script to obtain and save the Cramér-Rao Lower Bound (CRLB)-based 
% protocol. This script should be the first to run. To run the script, it's
% necessary to predefine the following parameters:
% - "N": number of sampled sets of measurements, i.e., of a group of a
% diffusion gradient direction (x,y,z), b-value, TD and TI. Example: 500.
% MODIFY IN LINE 49
% - "path_samples": file with the Diffusion gradient directions for a 
% specified number of directions. For instance, the method developed by 
% Caruyer et al. (Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-
% 1540) can be used to define the directions. Example: 'samples500.txt'.
% MODIFY IN LINE 61
%
% The CRLB-based protocol is finally saved using the format
% 'protocolN.mat'. If another name is prefered, MODIFY IN LINE 106
%
% Authors: Chantal Tax & Alvaro Planchuelo-Gomez, version 28 May 2024

clear;
%% Derive equations
% syms S bval TI TR TD gx gy gz Theta Phi Dpar Dper S0 T2s T1 real positive 
% x = [Theta,Phi,Dpar,Dper,T2s,T1,S0];
% b_delta = 1;
% S = simplify(x(7) .* exp(b_delta ./ 3.0 .* bval .* (x(3) - x(4)) - bval ./ 3.0 .* (x(3) + 2.0 .* x(4)) - bval .* b_delta .* ((dot([gx gy gz],[cos(x(2))*sin(x(1)),sin(x(1))*sin(x(2)),cos(x(1))],2)).^2 .* (x(3) - x(4)))) ...
%             .* abs(1.0 - 2.0 .* exp(-TI./x(6)) + exp(-TR./x(6))) .* exp(-TD./x(5))); %updated equation x(3) + 2.0 .* x(4) in the second term instead of x(4) + 2.0 .* x(3) as diffusivities were underestimated
% 
% dSdTheta = simplify(diff(S,Theta));
% dSdPhi = simplify(diff(S,Phi));
% dSdDpar = simplify(diff(S,Dpar));
% dSdDper = simplify(diff(S,Dper));
% dSdT2s = simplify(diff(S,T2s));
% dSdT1 = simplify(diff(S,T1));
% dSdS0 = simplify(diff(S,S0));

%% Set up tissue parameters (updated; original values for the paper 0.5, 0.5, 2, 1, 4, 1.5, 1, preserved SNR and TR)
Theta = 1.56; Phi = 1.88; Dpar = 0.81; Dper = 0.58; T2s = 7.2; T1 = 1.159; S0 = 2.37; % scale everything around 1; D in mum2/ms, T2* in 10^1 ms, T1 in s
SNR = 30; sigma = S0/SNR;
TR = 7.5; 

%% Test 1
% g = [1 0 0;
%     0 1 0;
%     0 0 1];
% TI = [0.01 0.1 1]';
% TD = [0 0.02 0.03]';
% bval = [0 1 2]';
% F = CRLB(Theta,Phi,Dpar,Dper,T2s,T1,S0,bval,TD,TI,TR,g,sigma)

%% Load protocol
N = 500; % REPLACE BY THE DESIRED NUMBER OF MEASUREMENTS
opts = delimitedTextImportOptions("NumVariables", 4);
opts.DataLines = [23, 23+N-1];
opts.Delimiter = "\t";
opts.VariableNames = ["Var1", "u_x", "u_y", "u_z"];
opts.SelectedVariableNames = ["u_x", "u_y", "u_z"];
opts.VariableTypes = ["string", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");
opts = setvaropts(opts, ["u_x", "u_y", "u_z"], "ThousandsSeparator", ",");
path_samples = 'samples500_txt'; % WRITE HERE PERSONAL PATH INSTEAD
samples = readtable(path_samples, opts);
g = table2array(samples);

%% Test 2
% shellsize = 3;
% nmeas = size(g,1);
% m = mod(nmeas,shellsize);
% nshells = (nmeas-m)/shellsize+1;
% bval = linspace(0,3,nshells)';
% TD = linspace(0,0.05,nshells)';
% TI = linspace(0,7,nshells)';
% x = [bval;TD;TI];
% F = loss(Theta,Phi,Dpar,Dper,T2s,T1,S0,x,TR,g,sigma);

%% Run optimisation
shellsize = 3;
nmeas = size(g,1);
m = mod(nmeas,shellsize);
nshells = (nmeas-m)/shellsize+1;

lb = zeros(nshells,3); lb = lb(:);
ub = [3.05*ones(nshells,1) 5.5*ones(nshells,1) 7.35*ones(nshells,1)]; ub = ub(:);
niter = 30;
options = optimoptions('lsqnonlin','TolFun',1e-8,'TolX',1e-8,'MaxIter', 1000, 'MaxFunEvals', 100000, 'Display','off');

xall = zeros([nshells 3 niter]);
resnormall = zeros([1 niter]);

parfor iter = 1:niter
    x0 = rand(size(ub)).*ub;
    [x,resnorm] = lsqnonlin(@(x)loss(Theta,Phi,Dpar,Dper,T2s,T1,S0,x,TR,g,sigma),x0,lb,ub,options);
    resnormall(iter) = resnorm;
    xall(:,:,iter) = reshape(x,nshells,3);
end

[~,I] = min(resnormall);
xall_ = zeros(size(xall(:,:,1)));
xall_(:,:) = xall(:,:,I);

bval = repmat(xall_(:,1),[1 shellsize])'; bval = bval(:); bval = bval(1:nmeas);
TD = repmat(xall_(:,2),[1 shellsize])'; TD = TD(:); TD = TD(1:nmeas);
TI = repmat(xall_(:,3),[1 shellsize])'; TI = TI(:); TI = TI(1:nmeas);
protocol = [g bval TD TI];
% We save the CRLB-based protocol with the format 'protocolN.mat'
save(['protocol' num2str(N) '.mat'])

% Function to compute the loss
function F = loss(Theta,Phi,Dpar,Dper,T2s,T1,S0,x,TR,g,sigma)
nshells = size(x,1)/3;
nmeas = size(g,1);
m = mod(nmeas,nshells-1);
shellsize = (nmeas-m)/(nshells-1);
bval = x(1:nshells); bval = repmat(bval,[1 shellsize])'; bval = bval(:); bval = bval(1:nmeas);
TD = x((nshells+1):2*nshells); TD = repmat(TD,[1 shellsize])'; TD = TD(:); TD = TD(1:nmeas);
TI = x((2*nshells+1):3*nshells); TI = repmat(TI,[1 shellsize])'; TI = TI(:); TI = TI(1:nmeas);
F = CRLB(Theta,Phi,Dpar,Dper,T2s,T1,S0,bval,TD,TI,TR,g,sigma);
end

% Function to implement the CRLB-based optimisation using the Jacobian
% matrix (updated derivatives according to new equation as shown in line 23)
function F = CRLB(Theta,Phi,Dpar,Dper,T2s,T1,S0,bval,TD,TI,TR,g,sigma)
gx = g(:,1); gy = g(:,2); gz = g(:,3);
dSdPhi = double(-2.*S0.*bval.*exp((bval.*(Dpar - Dper))./3 - (bval.*(Dpar + 2.*Dper))./3 - bval.*(Dpar - Dper).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2).*exp(-TD./T2s).*abs(exp(-TR./T1) - 2.*exp(-TI./T1) + 1).*(Dpar - Dper).*(gy.*cos(Phi).*sin(Theta) - gx.*sin(Phi).*sin(Theta)).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)));
dSdDpar = double(-S0.*bval.*exp((bval.*(Dpar - Dper))./3 - (bval.*(Dpar + 2.*Dper))./3 - bval.*(Dpar - Dper).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2).*exp(-TD./T2s).*abs(exp(-TR./T1) - 2.*exp(-TI./T1) + 1).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2);
dSdDper = double(-S0.*exp((bval.*(Dpar - Dper))./3 - (bval.*(Dpar + 2.*Dper))./3 - bval.*(Dpar - Dper).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2).*exp(-TD./T2s).*abs(exp(-TR./T1) - 2.*exp(-TI./T1) + 1).*(bval - bval.*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2));
dSdT2s = double((S0.*TD.*exp((bval.*(Dpar - Dper))./3 - (bval.*(Dpar + 2.*Dper))./3 - bval.*(Dpar - Dper).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2).*exp(-TD./T2s).*abs(exp(-TR./T1) - 2.*exp(-TI./T1) + 1))./T2s.^2);
dSdT1 = double(-S0.*exp((bval.*(Dpar - Dper))./3 - (bval.*(Dpar + 2.*Dper))./3 - bval.*(Dpar - Dper).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2).*exp(-TD./T2s).*sign(exp(-TR./T1) - 2.*exp(-TI./T1) + 1).*((2.*TI.*exp(-TI./T1))./T1.^2 - (TR.*exp(-TR./T1))./T1.^2));
dSdS0 = double(exp((bval.*(Dpar - Dper))./3 - (bval.*(Dpar + 2.*Dper))./3 - bval.*(Dpar - Dper).*(gz.*cos(Theta) + gx.*cos(Phi).*sin(Theta) + gy.*sin(Phi).*sin(Theta)).^2).*exp(-TD./T2s).*abs(exp(-TR./T1) - 2.*exp(-TI./T1) + 1));

J = 1/sigma.^2*[dSdTheta'*dSdTheta, dSdPhi'*dSdTheta, dSdDpar'*dSdTheta, dSdDper'*dSdTheta, dSdT2s'*dSdTheta, dSdT1'*dSdTheta, dSdS0'*dSdTheta;
    dSdPhi'*dSdTheta,     dSdPhi'*dSdPhi,   dSdDpar'*dSdPhi,   dSdDper'*dSdPhi,   dSdPhi'*dSdT2s,   dSdPhi'*dSdT1,   dSdPhi'*dSdS0;
    dSdDpar'*dSdTheta,  dSdDpar'*dSdPhi,     dSdDpar'*dSdDpar,  dSdDpar'*dSdDper,  dSdDpar'*dSdT2s,  dSdDpar'*dSdT1,  dSdDpar'*dSdS0;
    dSdDper'*dSdTheta,  dSdDper'*dSdPhi,  dSdDpar'*dSdDper,     dSdDper'*dSdDper,  dSdDper'*dSdT2s,  dSdDper'*dSdT1,  dSdDper'*dSdS0;
    dSdT2s'*dSdTheta,   dSdPhi'*dSdT2s,   dSdDpar'*dSdT2s,   dSdDper'*dSdT2s,     dSdT2s'*dSdT2s,   dSdT1'*dSdT2s,   dSdS0'*dSdT2s;
    dSdT1'*dSdTheta,    dSdPhi'*dSdT1,    dSdDpar'*dSdT1,    dSdDper'*dSdT1,    dSdT1'*dSdT2s,     dSdT1'*dSdT1,    dSdS0'*dSdT1;
    dSdS0'*dSdTheta,    dSdPhi'*dSdS0,    dSdDpar'*dSdS0,    dSdDper'*dSdS0,    dSdS0'*dSdT2s,    dSdS0'*dSdT1,     dSdS0'*dSdS0];
Jinv = inv(J);
F = sum(diag(Jinv)./([Theta,Phi,Dpar,Dper,T2s,T1,S0].^2)');
end
