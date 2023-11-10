function [assignment,cost] = subselect_CRLB(N,protocol_file,parameters_file)
% Function to obtain the selected measurements via the Cramer-Rao Lower
% Bound (CRLB)-based optimisation. A set of measurements is taken from a
% prespecified acquisition protocol by comparing the CRLB-based protocol
% with the prespecifed one and taking the most similar specifications. This
% function should be run after "T1T2sD_CRLB.m"
%
% Inputs:
% -N: number of selected measurements (e.g., 500)
% -protocol_file: path to the .mat file with the information from the
% CRLB-based protocol (e.g., 'protocol500.mat')
% -parameters_file: path to the .txt file (or equivalent format) with the
% diverse original MRI acquisition parameters in the following order for
% each row: diffusion gradient orientation (x,y,z; first three values),
% b-value (s/mm^2), TI (ms), TE (ms). Example: 'parameters_new.txt'
% 
% Outputs:
% -assignment: the values of the rows that represent the selected
% measurements, i.e., the closest set of parameters from the original 
% acquisition protocol to the CRLB-based protocol. Note: the returned
% values are in Python format, i.e., first file = 0
% -cost: minimum cost based on the assignment (read documentation of
% munkres.m for higher details)
%
% Authors: Chantal Tax & Alvaro Planchuelo-Gomez, version 10 Novemeber 2023

% Load the protocol: format of file name: load(['protocol' num2str(N) '.mat']);
load(protocol_file);

% We read the acquisition parameters and adjust its values to match those
% from the CRLB-based protocol
% p = dlmread('parameters_new.txt');
p = dlmread(parameters_file);
p(:,4) = p(:,4)/1000; % bval
p(:,5) = p(:,5)/1000; % TI
p(:,6) = (p(:,6)-min(p(:,6)))/10; %TD

% The original and CRLB-based protocols are organised and the Hungarian
% algorithm is employed to obtain the final results
pMUDI = repmat(p(:,4:6),[1 1 N]); % Name based on the original MUDI protocol used to implement the code
pCRLB = repmat(protocol(:,[4 6 5]),[1 1 size(p,1)]); % bval, TD, TI; protocol is a variable obtained from 'protocol_file'
pCRLBt = permute(pCRLB,[3 2 1]);%[size(p,1) 3 N]);
costmat = squeeze(sqrt((pMUDI(:,1,:)-pCRLBt(:,1,:)).^2+(pMUDI(:,2,:)-pCRLBt(:,2,:)).^2+(pMUDI(:,3,:)-pCRLBt(:,3,:)).^2));
[assignment,cost] = munkres(costmat'); % Hungarian algorithm
assignment = assignment-1; % To match the main code in Python (first index = 0)
%disp(assignment) % to display the row values if desired
end