clear all
close all
str = '11-13-16';
data_folder = 'H:\processed_file_T220_94GHz\Staring';
addpath(data_folder)
load(['Range_Time_Intensity_plot_T-220_2024-03-01_' str '.mat'])
range_bins=1024;
chirp_period=77.269e-6;
B=(62.5 * 12)* 1e6;
pf=1;
c=3e8;
range=c/(2*B*pf):c/(2*B*pf):(c/(2*B*pf))*(range_bins/2);
figure(1)
colormap(jet(256))
imagesc([0,chirp_period*length(spec)],range,spec);
clim([-90,0])
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