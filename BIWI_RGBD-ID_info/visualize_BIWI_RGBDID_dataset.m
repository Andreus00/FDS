%% Visualize data of the BIWI RGBD-ID Dataset from the chosen person and from training or testing sets %%
% It supposes the folder structure to be the following:
%   base_dir -> Training -> 000 
%                           ... 
%                           049
%            -> Testing -> Still -> 000 
%                                   ... 
%                                   049
%                       -> Walking -> 000 
%                                     ... 
%                                     049
%
% If you use this dataset, please cite:
% M. Munaro, A. Fossati, A. Basso, E. Menegatti and L. Van Gool. 
% One-Shot Person Re-Identification with a Consumer Depth Camera, 
% Book Chapter in "Person Re-Identification", Springer, 2013.
%
% Author: Matteo Munaro
% Date: 27th July 2013

addpath(pwd)
clc; clear all; close all;

%% Parameters to choose:
base_dir = uigetdir(pwd); % select path to the main folder (containing 'Training' and 'Testing')
dataset = 0;  % 0: training set, 1: still testing set, 2: walking testing set;
person_number = 0;  % person to be visualized (ranges between 0 and 49)

%% Check parameters:
number_string = num2str(person_number, '%03d');
if (dataset == 0)
    data_dir = [base_dir '/Training/' number_string];
else if (dataset == 1)
        data_dir = [base_dir '/Testing/Still/' number_string];
    else if (dataset == 2)
        data_dir = [base_dir '/Testing/Walking/' number_string];
        else
            display('ERROR: not valid ''dataset'' value!')
            return;
        end
    end
end

%% Read filenames:
if ~exist(data_dir, 'dir'); % if the data directory does not exist
    display('ERROR: the chosen person is not present in the chosen dataset.');
    return;
end

cd(data_dir)
rgb_files = dir('*b.jpg');
usermap_files = dir('*p.pgm');
depth_files = dir('*h.pgm');
skeleton_files = dir('*l.txt');
ground_files = dir('*f.txt');

% Parameters for skeleton visualization:
line_width = 3;
scale_factor_skelToImage = 1;
plot_only_tracked_joints = true;

%% Main loop:
h = figure;
set(h,'Position',[0 0 2600 480]);
for i = 1:size(rgb_files,1)
   clf;
     
   % Visualize rgb image:
   subplot(1,3,1);
   imshow(imread(rgb_files(i).name));
   
   % Visualize depth image:
   subplot(1,3,2);
   imagesc(imread(depth_files(i).name));
   
   % Visualize user map:
   subplot(1,3,3);
   imagesc(imread(usermap_files(i).name));
   
   % Plot skeleton on the depth image:
   Pos = dlmread(skeleton_files(i).name);   
   if Pos(1,1) > 0  % if there is a skeleton
       subplot(1,3,2);
       plotSkeleton2D(Pos(1:20,:), 0, line_width, false, scale_factor_skelToImage, plot_only_tracked_joints);
   end
   
   % Read and display ground coefficients:
   ground = dlmread(ground_files(i).name);
   if (sum(abs(ground)) > 0)
       disp(ground);
   end
   
   pause(eps);
end

