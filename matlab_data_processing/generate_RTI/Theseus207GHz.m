close all
clear all


data_folder = '/media/zg34/TOSHIBA EXT/processed_raw_data_207GHz';
save_folder = '/media/zg34/TOSHIBA EXT/processed_file_Theseus_207GHz';
datFiles = dir(fullfile(data_folder, '*.mat'));
fileNames = {datFiles.name};
% fileNames = fileNames(6:numel(fileNames));
chunk_size = 4096000;
for i=1:length(fileNames)
    file_name = fileNames(i);
    file_name = file_name{1};
    str = file_name(1:19);
    disp(i);
    range_time_intensity(data_folder,save_folder,str,chunk_size);
end

function range_time_intensity(data_foldername,save_foldername,str,chunk_size)
addpath(data_foldername)
range_bins=4096; %num of total samples per chirp
chirp_period=67.58e-6;
c=3e8;
B= 2000* 1e6; %%Hz
pf=1; %pad factor

matObj = matfile([str '_207GHz_FMCW.mat']);
totalElements = size(matObj,"data",1);
startPos = 1;
spec = [];
while startPos <= totalElements
    endPos = min(startPos + chunk_size - 1, totalElements);
    dataChunk = matObj.data(startPos:endPos, 1);
    s=length(dataChunk)/range_bins;
    AA =reshape(dataChunk,[range_bins,s]);
    clear dataChunk
    windowmesh = meshgrid(blackmanharris(range_bins),1:s)';
    Win_corr = 0.3587; 
    dataout_wind =windowmesh.*AA;
    spectrum =fft(dataout_wind,[],1); %fft down the columns of the matrix
    spectrum2 =spectrum(1:length(spectrum(:,1))/2,:);
    correction_dB =  -3 -10*log10(50) +30 -20*log10(range_bins/2) -20*log10(Win_corr)-90.3090;
    spec_temp = 20*log10(abs(spectrum2))+correction_dB;
    spec = [spec spec_temp];
    startPos = endPos + 1;
end

range=c/(2*B*pf):c/(2*B*pf):(c/(2*B*pf))*(range_bins/2);




figure(1)
colormap(jet(256))
imagesc([0,chirp_period*length(spec)],range,spec);
clim([-80,-30])
xlabel('Time (s)')
ylabel('Range (m)')
%title('Spectrogram')
axis xy
%ylim([0 30])
set(gca,'FontWeight','Bold','FontSize',40)
cc=colorbar;
ylabel(cc,'Signal strength (dBm)')
set(gcf,'color','white')
set(gcf,'units','normalized','outerposition',[0 0 1 1])
set(gcf,'WindowState','maximized')
drawnow

saveas(gcf,fullfile(save_foldername,['Range_Time_Intensity_plot_Theseus_' str '.png']));
close all
disp("finish saving fig");
save(fullfile(save_foldername,['Range_Time_Intensity_plot_Theseus_' str '.mat']),"spec");
disp("finish saving mat file")
end