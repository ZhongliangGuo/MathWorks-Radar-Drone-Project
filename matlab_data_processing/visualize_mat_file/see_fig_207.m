clear all
close all
str = '10-24-23';
data_folder = 'G:\processed_file_Theseus_207GHz';
addpath(data_folder)
% load(['Range_Time_Intensity_plot_Theseus_2024-03-01_' str '.mat'])
data = matfile(['Range_Time_Intensity_plot_Theseus_2024-02-06_' str '.mat']);
subset = data.spec;
range_bins=4096;
chirp_period=67.58e-6;
c=3e8;
B= 2000* 1e6; %%Hz
pf=1; %pad factor
range=c/(2*B*pf):c/(2*B*pf):(c/(2*B*pf))*(range_bins/2);

figure(1)
colormap(jet(256))
imagesc([0,chirp_period*length(subset)],range,subset);
clim([-80,-30])
xlabel('Time (s)')
ylabel('Range (m)')
axis xy
set(gca,'FontWeight','Bold','FontSize',40)
cc=colorbar;
ylabel(cc,'Signal strength (dBm)')
set(gcf,'color','white')
set(gcf,'units','normalized','outerposition',[0 0 1 1])
set(gcf,'WindowState','maximized')
title(str);
drawnow