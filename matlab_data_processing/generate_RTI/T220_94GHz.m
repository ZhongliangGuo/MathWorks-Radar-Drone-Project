data_folder = 'G:\MathWorks_Bird_Trial_2024-02-06_T-220';
save_folder = 'G:\processed_file_T-220_94GHz';
datFiles = dir(fullfile(data_folder, '*.dat'));
fileNames = {datFiles.name};
% fileNames = fileNames(1:4);
chunk_size = 2048000;
for i=1:length(fileNames)
    file_name = fileNames(i);
    file_name = file_name{1};
    str = file_name(1:19);
    range_time_intensity(data_folder,save_folder,str,chunk_size);
    disp(i);
end

function range_time_intensity(data_foldername,save_foldername,str,chunk_size)
addpath(data_foldername)
fh = fopen([str '_94GHz_FMCW.dat'],'r');
if fh == -1
    error('File open failed.');
end
spec = [];
chirp_period = 77.269e-6;
while ~feof(fh)
    data = fread(fh, chunk_size,'uint16');
    data=data-32768;
    data11=data(1:2:end);% - 32768;
    data22=data(2:2:end);% - 32768;
    data1 = complex(data11, data22);
    range_bins=1024;
    s=floor((length(data1)/range_bins)/1);
    data_segment=data1(1:range_bins*s);
    AA =reshape(data_segment,[range_bins,s]);
    c=3e8;
    B=(62.5 * 12)* 1e6; %%Hz
    pf=1; %pad factor
    windowmesh = meshgrid(blackmanharris(range_bins),1:s)';
    Win_corr = 0.3587;
    dataout_wind =windowmesh.*AA;
    PL =range_bins*pf; %Pad length
    %zero-phase padding. Chop the windowed signal and half. Take the second
    %half of the signal and place it at the start. Fill the middle of the
    %data set with zeros.
    dataout_P =padarray(dataout_wind,PL-range_bins,'post');
    dataout_P =circshift(dataout_P,[((PL-range_bins)+(range_bins/2)),0]);
    spectrum =fft(dataout_P,[],1); %fft down the columns of the matrix
    spectrum2 =spectrum(1:length(spectrum(:,1))/2,:);
    correction_dB =  -3 -10*log10(50) +30 -20*log10(range_bins/2) -20*log10(Win_corr) -90.3090;
    spec =[spec (20*log10(abs(spectrum2))+correction_dB)];  %+ a correction to the power
end

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
drawnow
saveas(gcf,fullfile(save_foldername,['Range_Time_Intensity_plot_T-220_' str '.png']));
save(fullfile(save_foldername,['Range_Time_Intensity_plot_T-220_' str '.mat']),"spec");
% saveas(gcf,fullfile(save_foldername,['Range_Time_Intensity_plot_T220_' str '.fig']));
close all
end