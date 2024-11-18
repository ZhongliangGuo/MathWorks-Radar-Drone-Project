data_folder = '/media/zg34/TOSHIBA EXT/MathWorks_Drone_Trial_2024-03-01_T-220/CW';
save_folder = '/media/zg34/TOSHIBA EXT/processed_file_T220_94GHz/CW';
datFiles = dir(fullfile(data_folder, '*.dat'));
fileNames = {datFiles.name};
fileNames = fileNames(3:numel(fileNames));
for i=1:length(fileNames)
    file_name = fileNames(i);
    file_name = file_name{1};
    str = file_name(1:19);
    spectrogram(data_folder,save_folder,str);
    disp(i);
end

function spectrogram(data_foldername,save_foldername,str)
fh = fopen(fullfile(data_foldername,[str,'_CW.dat']),'r');
data = fread(fh, inf, 'double');
ss=length(data)/131072;
% spectrogram=zeros(ss,512,255);
% spectrogram=zeros(ss,512,65025);
v_real=zeros(ss,65536);
v_im=zeros(ss,65536);
spectrum=zeros(ss,65536);
x_spec=zeros(1,65536);
Fs=200e3;
freq=-Fs/2:Fs/65536:(Fs/2)-(Fs/65536);
time=linspace(0,65536/Fs,65536);
corr =-3-(10*log10(50))+30-(20*log10(0.5))-20*log10(65536);
corr_stft =-3-(10*log10(50))+30-(20*log10(0.0782))-20*log10(512);
vid = VideoWriter(fullfile(save_foldername,str));
vid.FrameRate=4;
open(vid);
for n=1:ss
    data1= data((131072*(n-1))+1:(131072*(n-1))+131072,1);
    data1=data1';
    v_real(n,:)=data1(1:2:end); % channel 1
    v_im(n,:)=data1(2:2:end); % channel 2
    x = complex(data1(1:2:end), data1(2:2:end));
    Vr=x;
    win=hanning(65536);

    for nn=1:65536
        x_spec(nn)=x(nn)*win(nn);
    end
    spectrum(n,:)=mag2db(fftshift(abs(fft(x_spec)))) + corr;
    np = length(Vr);

    wd = 512;
    ns=np-(wd-1);
    TF=zeros(wd*1,ns);
    win=gausswin(wd,16);
    sum=zeros(wd*1,ns);
    for m= 1:1
        for k=1:ns
            tmp=win' .* x(m,k:k+wd-1);
            tmp=fft(tmp);
            TF(:,k)=tmp(1:length(tmp));
        end
        sum=sum + TF;
    end
    sumdB= 20*log10(fftshift(abs(sum),1))+corr_stft;
    % draw sumdB
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    drawnow
    colormap(jet(256))
    imagesc(time,freq/1000,sumdB)
    title(['Spectrogram' ' ' num2str(n)])
    ylabel('Doppler (kHz)')
    xlabel('Time(sec)')
    clim([-70,-10])
    axis xy
    cc=colorbar;
    ylabel(cc,'Signal strength (dBm)')
    set(gca,'FontWeight','Bold','FontSize',30)
    set(gcf,'color','white')
    frame = getframe(gcf);
    writeVideo(vid,frame);
    pause(1.0)
end
close(vid);
close all
end