close all
clear all
% some constants
chirp_period=67.58e-6;
range_bins=4096;
lambda=0.0014;
range_resolution=0.075; % m

% hyperparameters
%     for signal strength
strength_from=-60;
strength_abs=50;
%     for FFT window
window_length=256;
width_factor=16;
%    for time interval
time_interval=0.25; % sec
time_overlap=0.1; % sec

% file system stuffs
data_folder = 'G:\processed_raw_data_207GHz';
save_folder = fullfile('H:\extra_training_data\207GHz\noise',['interval_' num2str(time_interval) '_wd_' num2str(window_length)]);
label_path = 'noise207GHz.csv';

% calculate the data points for each image
num_points_interval=ceil(time_interval/chirp_period)*range_bins; % int
num_points_overlap=ceil(time_overlap/chirp_period)*range_bins; % int
% use the pre-defined csv file to calculate the range_start and range_stop
opts = detectImportOptions(label_path);
opts=setvartype(opts,'char');
opts.Delimiter = ',';
A = readtable(label_path,opts);  % the table generated from the csv file
for i=1:size(A,1)
% for i=42:42
    file_name = table2cell(A(i,1));
    file_name = file_name{1};
    str = file_name(1:19);
    range_start=table2cell(A(i,2));
    range_start=str2double(range_start{1});
    range_stop=table2cell(A(i,3));
    range_stop=str2double(range_stop{1});
    range_start=floor(range_start/range_resolution);
    range_stop = ceil(range_stop/range_resolution);
    disp(i);
    gen_spectrogram( ...
        data_folder, ...
        save_folder, ...
        str, ...
        num_points_interval, ...
        num_points_overlap, ...
        range_start, ...
        range_stop, ...
        range_bins, ...
        chirp_period, ...
        lambda, ...
        strength_from, ...
        strength_abs, ...
        window_length, ...
        width_factor);
end
function gen_spectrogram( ...
    data_foldername, ...
    save_foldername, ...
    str, ...
    interval, ...
    overlap, ...
    range_start, ...
    range_stop, ...
    range_bins, ...
    chirp_period, ...
    lambda, ...
    strength_from, ...
    strength_abs, ...
    window_length, ...
    width_factor)
matObj = matfile(fullfile(data_foldername,[str '_207GHz_FMCW.mat']));
total_num_points=size(matObj,"data",1);
positions=1:(interval-overlap):(total_num_points-interval+1);
if positions(numel(positions))+interval-1<total_num_points
    positions=positions(1:(numel(positions)-1));
end
mkdir(fullfile(save_foldername,str));
i=0;
for start_pos=positions
    try
        data_chunk=matObj.data(start_pos:(start_pos+interval-1),1);
    catch
        disp([str ' failed to make image ' num2str(i) ' with start position ' num2str(start_pos)])
        continue
    end
    AA=reshape(data_chunk,[range_bins,interval/range_bins]);
    clear data_chunk
    Fs=1/chirp_period;
    freq=-Fs/2:Fs/range_bins:(Fs/2)-(Fs/range_bins);
    velocity=-((freq*lambda)/2);
    s=interval/range_bins;
    windowmesh = meshgrid(blackmanharris(range_bins),1:s)';
    Win_corr = 0.3587;
    dataout_wind =windowmesh.*AA;
    PL =range_bins; %Pad length
    dataout_P =padarray(dataout_wind,PL-range_bins,'post');
    dataout_P =circshift(dataout_P,[((PL-range_bins)+(range_bins/2)),0]);
    spectrum =fft(dataout_P,[],1); %fft down the columns of the matrix
    Vr=(spectrum(range_start:range_stop,:));
    x=(Vr);
    np = length(Vr);
    ns=np-(window_length-1);
    TF=zeros(window_length*1,ns);
    win=gausswin(window_length,width_factor);
    Win_corr_slow=mean(win);
    correction_dB =  -3 -10*log10(50) +30 -20*log10(range_bins/2) -20*log10(Win_corr) -90.3090;
    correction_dB_Slow = -20*log10(window_length) -20*log10(Win_corr_slow) + correction_dB;
    x_i=0;
    for m=1:(range_stop-range_start)+1
        x_i=x_i+x(m,:);
    end

    cpi=64;
    phase_unwrapped=unwrap(angle(x_i));
    tchirp=length(phase_unwrapped)/cpi;
    a1=1:5;
    for n=1:tchirp-1
        a2=phase_unwrapped(((n*cpi)-4):(n*cpi));
        a3=interp1(a1,a2,[6,7,8],'linear','extrap');
        diff=a3(1)-phase_unwrapped((n*cpi)+3);
        phase_unwrapped((n*cpi)+1:(n*cpi)+3)=a3;
        phase_unwrapped((n*cpi)+4:length(phase_unwrapped))= phase_unwrapped((n*cpi)+4:length(phase_unwrapped))+diff;
    end
    x_i_real=abs(x_i) .* cos(phase_unwrapped);
    x_i_imag=abs(x_i) .* sin(phase_unwrapped);
    x_i_int=complex(x_i_real,x_i_imag);
    sum=zeros(window_length*1,ns);
    for m= 1:1
        for k=1:ns
            tmp=win' .* (x_i_int(m,k:k+window_length-1)); %windowing and overlapping
            tmp=fft(tmp);
            TF(:,k)=tmp(1:length(tmp));
        end
        sum=sum + TF;
    end
    sumdB= 20*log10(fftshift(abs(sum),1)+eps) + correction_dB_Slow;
    fig = figure('Visible', 'off');
    colormap(jet(256))
    imagesc([0,chirp_period*np],velocity, flipud(sumdB));
    clim([strength_from,strength_from+strength_abs])
    axis xy
    axis off
    axis tight
    set(gca, 'Position', [0 0 1 1]);
    set(gcf,'color','white')
    set(gcf,'units','pixels','position',[0 0 512 512])
    set(gcf,'WindowState','maximized')
    drawnow
    frame=getframe(gca);
    img = frame2im(frame);
    imwrite(img,fullfile(save_foldername,str,[int2str(i) '.png']));
    i=i+1;
    close(fig);
end
end