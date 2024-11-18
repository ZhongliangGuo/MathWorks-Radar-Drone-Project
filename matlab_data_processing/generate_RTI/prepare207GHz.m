clear all
close all
data_folder = 'G:\MathWorks_Bird_Trial_2024-02-06_Theseus';
save_folder = 'G:\processed_raw_data_207GHz';

datFiles = dir(fullfile(data_folder, '*.dat'));
fileNames = {datFiles.name};
% fileNames = fileNames(27:60);
for i=11:length(fileNames)
    file_name = fileNames(i);
    file_name = file_name{1};
    str = file_name(1:19);
    prepare(data_folder,save_folder,str);
    disp(i);
end

function prepare(data_foldername,save_foldername,str)
% addpath(data_foldername)
fh = fopen(fullfile(data_foldername,[str '_207GHz_FMCW.dat']),'r');
if fh == -1
    error('File open failed.');
end
matObj = matfile(fullfile(save_foldername, [str '_207GHz_FMCW.mat']), 'Writable', true);
% try
    data_all = fread(fh, inf, 'short');
% catch
    % disp([str ' failed'])
    % return;
% end
data=zeros((length(data_all)-((length(data_all)/4224))*128),1); %spectrum adc card has 128 pretrigger data points, whcih are discarded
counter=1;counter_data=1;
for n=1:length(data_all)
    if n>128+(4224*(counter-1))
        data(counter_data,1)=data_all(n,1);
        counter_data=counter_data+1;
    end
    if n>1 && mod(n,4224)==0
        counter=counter+1;
    end
end
matObj.data=data;
fclose(fh);
clear matObj
end