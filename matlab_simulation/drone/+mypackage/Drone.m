classdef Drone<handle
    % This class is to simulate a drone object for CW radar detection
    % This class can create a drone object with certain rotors and blades,
    % then user can input any legal parameter combinations to see the
    % reaction from radar side with any legal spec.
    properties (Constant)
        c=2.99792458e8;  % light speed
        theta=0;  % offset
        Sigma=1;  % coefficient
        ck=0.200;  % coefficient for SUAV body
    end
    properties (SetAccess=private)
        i=sqrt(-1);  % imaginary unit
        NR;  % Number of rotors
        N;  % number of blades
        L1;  % distance from the blade root to the center of gravity
        L2;  % distance from the blade tip to the center of gravity
        d1;  % XXX
        NSample;  % the total number of samples during a simulation
        samplerate; % Number of sampling per second
        fc;  % carrier frequency, in GHz
    end
    methods
        function obj=Drone(NR,N,L1,L2,d1)
            % The constructor for the drone object.
            % NR is the number of rotors, it should be in range [2,4]
            % N is the number of blades, it should be in range [2,4]
            % L1 is the distance from the blade root to the drone gravity center
            % L2 is the distance from the blade tip to the drone gravity ceter
            % d1 is XXX
            assert(NR<=8 && NR>=2,"Number of rotors should be in range [2,8].");
            assert(N<=6 && N>=2,"Number of blades should be in range [2,6].");
            assert(L2>L1,"L2 should be bigger than L1.");
            obj.NR=NR;
            obj.N=N;
            obj.L1=L1;
            obj.L2=L2;
            obj.d1=d1;
        end
        function return_signal = generate_return_signal(obj,NSample,v,fc,beta,fr,R,samplerate,consider_drone_body)
            % This function will generate the radar return signal from the drone object.
            % NSample: the total number of samples during a simulation
            % v: the velocity of the drone object
            % fc: the carrier frequency (in GHz, 1GHz=1e9Hz) of radar
            % beta: the elevation angle (°), value range from 0° to 90°
            % fr: the revolution per second for each rotor, the length of
            % fr should be same as NR (number of rotors)
            % R: the range (m) from drone to radar
            % samplerate: the number of sampling per second
            % consider_drone_body: the simulation will consider the reflection from drone body or not
            % return: the signal radar received, in [1,NSample] shape
            assert(numel(fr)==obj.NR,"the length of fr should be same as NR (number of rotors).");
            obj.NSample=NSample;
            obj.samplerate=samplerate;
            obj.fc=fc;
            fc=fc*1e9;
            dt=1/samplerate;
            lambda=obj.c/fc;
            beta=beta*pi/180;
            return_signal = complex(zeros(1, NSample));
            for m=1:NSample
                t=dt*(m-1);
                summ=0;
                for qrn=1:obj.NR
                    for n=1:obj.N
                        nn=n-1;
                        Phi=(4*pi/lambda)*(sqrt((R+(v*t))^2+obj.d1^2-(2*(R+(v*t))*...
                            obj.d1*cos((obj.theta+(360/obj.NR*qrn))*pi/180)))+((obj.L1+obj.L2)/2)*...
                            cos(beta)* sin(2*pi.*fr(qrn)*t+(2*pi*nn)/obj.N));
                        AL=obj.Sigma*(obj.L2-obj.L1)*exp(obj.i*(2*pi*fc*t-Phi));
                        X=(4*pi/lambda)*((obj.L2-obj.L1)/2)*cos(beta)*sin(2*pi.*fr(qrn)*t+(2*pi*nn)/obj.N);
                        summ = summ + (AL*sinc(X/pi));
                    end
                    if consider_drone_body
                        Phi0 = (4*pi/lambda)*(sqrt((R+(v*t))^2+obj.d1^2-(2*(R+(v*t))*obj.d1*...
                            cos((obj.theta+(90* qrn))*pi/180)))+v*t+((obj.L1)/2)*cos(beta)*sin(2*pi*0*t+(2*pi*nn)/obj.N));
                        AL0 = obj.Sigma*(obj.ck)*exp(obj.i*(2*pi*fc*t-Phi0));
                        X0 = (4*pi/lambda)*((obj.L1)/2)*cos(beta)*sin(2*pi*0*t+(2*pi*nn)/obj.N);
                        summ = summ + (AL0*sinc(X0/pi));
                    end
                end
                return_signal(m) = summ;
            end
        end
        function [x_axis,freq_vector]=generate_doppler_spectrum(obj,return_signal)
            % The function to generate the doppler spectrum
            % return_signal: the output from generate_return_signal()
            % return: x_axis, the value for x axis
            %         freq_vector, the value for y axis
            freq_vector=abs(fftshift(fft(return_signal)));
            freq = -obj.samplerate/2:obj.samplerate/obj.NSample:obj.samplerate/2-obj.samplerate/obj.NSample;
            x_axis=-(freq*0.0032)/2;
        end
        function draw_doppler_spectrum(~,x_axis,freq_vector)
            figure
            plot(x_axis,freq_vector,'b')
            xlabel("frequencies")
            ylabel("Amplitude")
            drawnow
        end
        function [time_range,spectrogram]=generate_spectrogram(obj,return_signal,window_size)
            % This function is to generate spectrogram
            % return_signal: the output from generate_return_signal()
            % window_size: the size for scan window, recommend 64
            % return: time_range, the time domain range, for x axis
            %         spectrogram, the spectrogram, for y axis
            np=length(return_signal);
            ns=np-(window_size-1);
            TF=zeros(window_size*2,ns);
            win=gausswin(window_size,4);
            for k=1:ns
                tmp=win' .* return_signal(k:k+window_size-1); %windowing and overlapping
                tmp(length(tmp)+1:length(tmp)*2)=0; %zero padding
                tmp=fft(tmp);
                TF(:,k)=tmp(1:length(tmp));
            end
            time_range = [0,obj.NSample/obj.samplerate];
            spectrogram=20*log10(fftshift(abs(TF),1)+eps);
        end
        function draw_spectrogram(obj,time_range,spectrogram)
            figure
            colormap(jet(256))
            imagesc(time_range,[(-1/(2*(1/obj.samplerate))/1e3),(1/(2*(1/obj.samplerate))/1e3)],spectrogram)
            xlabel('Time (s)')
            ylabel('Doppler (KHz)')
            title(num2str(obj.fc)+" GHz")
            axis xy
            clim = get(gca,'CLim');
            set(gca,'CLim',clim(2) + [-80 0]);
            ylabel(colorbar,'Relative Signal strength (dB)')
            set(gca,'FontWeight','Bold','FontSize',10)
            set(gcf,'color','white')
            drawnow
        end
        function save_square_spectrogram(obj,time_range,spectrogram,img_path)
            fig = figure('Visible', 'off');
            colormap(jet(256))
            imagesc(time_range,[(-1/(2*(1/obj.samplerate))/1e3),(1/(2*(1/obj.samplerate))/1e3)],spectrogram)
            axis xy
            axis off
            axis tight
            clim = get(gca,'CLim');
            set(gca,'CLim',clim(2) + [-80 0]);
            set(gca, 'Position', [0 0 1 1]);
            set(gcf,'color','white')
            set(gcf,'units','pixels','position',[0 0 512 512])
            set(gcf,'WindowState','maximized')
            drawnow
            frame=getframe(gca);
            img = frame2im(frame);
            imwrite(img,img_path);
            close(fig);
        end
        function [time_range,HERM_lines]=generate_HERM_lines(obj,freq_vector)
            % This function is to generate the HERM (HElicopter Rotor Modulation) lines
            % freq_vector: the output from generate_doppler_spectrum()
            % return: time_range, the time domain range, for x axis
            %         HERM_lines, the HERM_lines, for y axis
            time_range=[0,obj.NSample/obj.samplerate];
            HERM_lines=20*log10(freq_vector');
        end
        function draw_HERM_lines(obj,time_range,HERM_lines)
            figure
            colormap(jet(256))
            imagesc(time_range,[-obj.samplerate/2,obj.samplerate/2],HERM_lines)
            xlim(time_range)
            xlabel('Time (s)')
            ylabel('Doppler (Hz)')
            title('HElicopter Rotor Modulation (HERM) lines')
            axis xy
            clim = get(gca,'CLim');
            set(gca,'CLim',clim(2) + [-50 0]);
            ylabel(colorbar,'Signal strength (dB)')
            set(gca,'FontSize',10)
            drawnow
        end
    end
end