data_folder = '/media/zg34/TOSHIBA EXT/MathWorks_Drone_Trial_2024-03-01_Blunderbuss';
save_folder = '/media/zg34/TOSHIBA EXT/processed_file_Blunderbuss_24GHz';
datFiles = dir(fullfile(data_folder, '*.dat'));
fileNames = {datFiles.name};
for i=1:length(fileNames)
    file_name = fileNames(i);
    file_name = file_name{1};
    str = file_name(1:19);
    range_time_intensity(data_folder,save_folder,str);
    disp(i);
end

function data = get_raw_signal(data_foldername,str)
addpath(data_foldername)
fh = fopen([str '_24GHz_FMCW.dat'],'r');
data = fread(fh, inf, 'uint16');
data = data(1:2:end) - 32768;
end

function range_time_intensity(data_foldername,save_foldername,str)
addpath(data_foldername)
fh = fopen([str '_24GHz_FMCW.dat'],'r');
data = fread(fh, inf, 'uint16');
data = data(1:2:end) - 32768;
% data2 = data(2:2:end) - 32768;
range_bins=512;
s=floor(length(data)/range_bins);
AA =reshape(data,[range_bins,s]);

chirp_period=200.00e-6;
c=3e8;
B=250e6;
pf=1;
range=c/(2*B*pf):c/(2*B*pf):(c/(2*B*pf))*(range_bins/2);

    windowmesh = meshgrid(blackmanharris(range_bins),1:s)';
    Win_corr = 0.3587;

dataout_wind =windowmesh.*AA;
PL =range_bins*pf;
dataout_P =padarray(dataout_wind,PL-range_bins,'post');
dataout_P =circshift(dataout_P,[((PL-range_bins)+(range_bins/2)),0]);
spectrum =fft(dataout_P,[],1); %fft down the columns of the matrix
spectrum2 =spectrum(1:length(spectrum(:,1))/2,:);
correction_dB =  -3 -10*log10(50) +30 -20*log10(range_bins/2) -20*log10(Win_corr) -90.3090;
spec = 20*log10(abs(spectrum2))+correction_dB;  %+ a correction to the power
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
drawnow

saveas(gcf,fullfile(save_foldername,['Range_Time_Intensity_plot_Blunderbuss_' str '.png']));
% saveas(gcf,fullfile(save_foldername,['Range_Time_Intensity_plot_T220_' str '.fig']));
close all
save(fullfile(save_foldername,['Range_Time_Intensity_plot_Blunderbuss_' str '.mat']),"spec");
end