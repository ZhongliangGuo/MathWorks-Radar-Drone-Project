clear all
close all
data_folder = 'G:\processed_file_Blunderbuss_24GHz';
str = '10-26-29';
addpath(data_folder)
chirp_period=200.00e-6;
B=250e6;
pf=1;
c=3e8;
range_bins=512;
range=c/(2*B*pf):c/(2*B*pf):(c/(2*B*pf))*(range_bins/2);
load(['Range_Time_Intensity_plot_Blunderbuss_2024-02-06_' str '.mat']);
figure(1)
colormap(jet(256))
imagesc([0,chirp_period*length(spec)],range,spec);
clim([-90,0])
xlabel('Time (s)')
ylabel('Range (m)')
title(str);
axis xy
set(gca,'FontWeight','Bold','FontSize',40)
cc=colorbar;
ylabel(cc,'Signal strength (dBm)')
set(gcf,'color','white')
set(gcf,'units','normalized','outerposition',[0 0 1 1])
drawnow