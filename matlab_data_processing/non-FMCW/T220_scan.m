data_folder = '/media/zg34/TOSHIBA EXT/MathWorks_Drone_Trial_2024-03-01_T-220/Scanning';
save_folder = '/media/zg34/TOSHIBA EXT/processed_file_T220_94GHz/Scanning/';
datFiles = dir(fullfile(data_folder, '*.dat'));
fileNames = {datFiles.name};
for i=1:length(fileNames)
    file_name = fileNames(i);
    file_name = file_name{1};
    str = file_name(1:19);
    range_time_intensity(data_folder,save_folder,str);
    disp(i);
end

function range_time_intensity(data_foldername,save_foldername,str)
addpath(data_foldername)
fh = fopen([str,'_FMCW.dat'],'r');
data = fread(fh, inf, 'uint16');
data11=data(1:2:end) - 32768;
data22=data(2:2:end) - 32768;
data1 = complex(data11, data22);

scan_angle=29.34; %degrees
%scan_angle=17.6344; %degrees
range_bins=2048;
nchirp=236;
%s=length(data1)/range_bins;
s=floor((length(data1)/range_bins)/1);
data_segment=data1(1:range_bins*s);
%chirps_per_scan=floor((scan_angle/60)/2.14e-3); %turntable pulse period is 2.13 ms
chirps_per_scan=236;
AA =reshape(data_segment,[range_bins,s]);
set_Win=1;

gap=8;

chirp_period=122.3434e-6;
Fs=1/chirp_period;
lambda=0.0032;
freq=-Fs/2:Fs/range_bins:(Fs/2)-(Fs/range_bins);
velocity=-((freq*lambda)/2);
c=3e8;
B=(62.5 * 12)* 1e6; %%Hz
pf=1; %pad factor
range=c/(2*B*pf):c/(2*B*pf):(c/(2*B*pf))*(range_bins/2);
deltaR=c/(2*B);
if set_Win == 1
    windowmesh = meshgrid(blackmanharris(range_bins),1:s)';
    Win_corr = 0.3587; 
elseif set_Win == 2
    windowmesh = meshgrid(rectwin(range_bins),1:s)';
    Win_corr = 1;
elseif set_Win == 3
    windowmesh = meshgrid(hann(range_bins),1:s)';
    Win_corr = 0.5;
elseif set_Win == 4
    windowmesh = meshgrid(blackman(range_bins),1:s)';
    Win_corr = 0.42;
elseif set_Win == 5
        windowmesh = meshgrid(flattopwin(range_bins),1:s)';
        Win_corr = 0.2156;
end

dataout_wind =windowmesh.*AA;

 

PL =range_bins*pf; %Pad length
%zero-phase padding. Chop the windowed signal and half. Take the second
%half of the signal and place it at the start. Fill the middle of the
%data set with zeros. 
dataout_P =padarray(dataout_wind,PL-range_bins,'post');
dataout_P =circshift(dataout_P,[((PL-range_bins)+(range_bins/2)),0]);
spectrum =fft(dataout_P,[],1); %fft down the columns of the matrix
spectrum2 =spectrum(1:length(spectrum(:,1))/2,:); 

win=gausswin(nchirp,4);
%win=flattopwin(nchirp);
Win_corr_slow=mean(win);
correction_dB =  -3 -10*log10(50) +30 -20*log10(range_bins/2) -20*log10(Win_corr) -90.3090;
correction_dB_Slow = -20*log10(nchirp) -20*log10(Win_corr_slow) + correction_dB;
spec = 20*log10(abs(spectrum2))+correction_dB;  %+ a correction to the power

%scan_angle=20; %degrees
%chirps_per_scan=floor((scan_angle/60)/2.13e-3); %turntable pulse period is 2.13 ms
num_of_frames=floor(length(spec(1,:))/chirps_per_scan);
tot_scan=zeros(num_of_frames,length(spec(:,1)),chirps_per_scan);
for n=1:num_of_frames
    if mod(n,2)==1
        tot_scan(n,:,:)=spec(:,((n-1)*chirps_per_scan)+1:1:((n-1)*chirps_per_scan)+chirps_per_scan);
    else
        tot_scan(n,:,:)=spec(:,((n)*chirps_per_scan):-1:((n-1)*chirps_per_scan)+1);
    end
end
     
%tot_scan=reshape(spec,[num_of_frames,length(spec(:,1)),chirps_per_scan]);

theta=((linspace(-(scan_angle/2),(scan_angle/2),chirps_per_scan-gap))*(pi/180));
%r= 1:range_bins/2;
 r=linspace(deltaR,(deltaR*range_bins/2),range_bins/2);
%r=linspace(deltaR*430,(deltaR*625),196);
%vid = VideoWriter('T-220_scan_test_random.avi');
vid = VideoWriter(fullfile(save_foldername,str));
vid.FrameRate=2;
%vid.FrameRate=0.85;
open(vid);
for n=1:num_of_frames
  %  if mod(n,2)==0
set(gcf,'units','normalized','outerposition',[0 0 1 1])

drawnow
colormap(jet(256))
  if (mod(n,2))==1
img=squeeze(tot_scan(n,1:512/1,1+gap:chirps_per_scan));
 else
     img=squeeze(tot_scan(n,1:512/1,1:chirps_per_scan-gap));
 end
    
% if n==1
%     background_ref=img;
%     img_sub=img;
% end
% if n>1
%     img_sub=img-background_ref;
% end
[xx,yy]=meshgrid(theta,r(1:512/1));
    [xxx,yyy]=pol2cart(xx,yy);
%     if mod(n,2) == 0
 %      if (n==17 || n==19)
%         surf(xxx,yyy,fliplr(img),'edgecolor','none');
%        else
    surf(xxx,yyy,fliplr(img),'edgecolor','none');
%        end
    %else
    %   surf(xxx,yyy,img,'edgecolor','none');
    %end
    colormap(jet)
    view(-90,90)
 %   pause(1.67)
%  imagesc(img)
grid minor
clim([-90 0])
xlabel('Y(metres)')
ylabel('X(metres)')
% title('PPI plot')
title(['Frame' ' ' num2str(n)])
%axis xy
set(gca,'FontWeight','Bold','FontSize',40)
cc=colorbar;
ylabel(cc,'Signal strength (dBm)')
set(gcf,'color','white')
daspect([1 1 1])
frame = getframe(gcf);
writeVideo(vid,frame);
pause(1.0)
end

close(vid);
close all
end