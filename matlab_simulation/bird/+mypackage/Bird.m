classdef Bird<handle

    properties (Constant)
        c=2.99792458e8;  % light speed
        % body
        Ba = 0.1;
        Bb = 0.1;
        Bc = 1.0;
        % wing
        Wa = 0.05;
        Wb = 0.05;
        Wc = 0.25;
    end

    properties (SetAccess=private)
        j = sqrt(-1);

        % radar paramerters
        % time smapling
        T = 1; % time duration
        nt = 15000; % number of samples
        % dt = T/nt; % time interval
        % ts = [0:dt:T-dt]; % time span
        dt = [];
        ts = [];

        lambda = 0.0032;   % wavelength of transmitted radar signal
        %         c = 2.99792458e8;
        % f0 = c/lambda;
        f0 = [];
        rangeres = 0.05;  % designed range resolution
        radarloc = [20, 0, -10];  % radar location
        % total number of range bins
        % nr = floor(2*sqrt(radarloc(1)^2+radarloc(2)^2+radarloc(3)^2)/rangeres);
        nr = [];

        % t = ts;
        t = [];
        F0 = 5.0; % flapping frequency
        V = 0.0; % forward translation velocity
        A1 = 40; % amplitude of flapping angel in degree
        ksi10 = 15; % lag flapping angle in degree
        L1 = 0.5; % length of segment 1
        % ksi1 = A1*cos(2*pi*F0*t)+ksi10; % flapping angle
        ksi1 = [];

        % segment 1 (upper arm)
        x1 = [];
        y1 = [];
        z1 = [];


        % segment 2 (forearm)
        A2 = 30; % amplitude of segment2 flapping angle
        ksi20 = 40; % lag flapping angle in degree
        L2 = 0.5; % length of segment 2
        C2 = 20; % amplitude of segment2 twisting angle


        ksi2 = [];
        theta2 = [];
        d = [];
        y2 = [];
        x2 = [];
        z2 = [];

        % bird body position
        Pb1 = [];
        Pb2 = [];
        Cb = [];

        % left upper arm position
        Pua11 = [];
        Pua21 = [];
        Puac1 = []; % upper arm center

        % left forearm position
        Pfa11 = [];
        Pfa21 = [];
        Pfac1 = []; % forearm center

        % right upper arm position
        Pua12 = [];
        Pua22 = [];
        Puac2 = []; % upper arm center

        % right forearm position
        Pfa12 = [];
        Pfa22 = [];
        Pfac2 = []; % forearm center

        % radar returns
        data = [];
        r_dist = [];
        distances = [];
        aspct = [];
        ThetaAngle = [];
        PhiAngle = [];
        rcs = [];
        amp = [];

        vx = 1; % forward translation velocity
        vy = 0;
        vz = 0;

    end

    methods
        function app = Bird(T, nt, fc, rangeres, radarloc, F0, A1, ksi10, L1, A2, ksi20, L2, C2, vx, vy, vz)
            app.T = T;
            app.nt = nt;
            app.lambda = 2.99792458e8/(fc*1e9);
            app.rangeres = rangeres;
            app.radarloc = radarloc;
            app.F0 = F0;
            app.A1 = A1;
            app.ksi10 = ksi10;
            app.L1 = L1;
            app.A2 = A2;
            app.ksi20 = ksi20;
            app.L2 = L2;
            app.C2 = C2;
            app.vx = vx;
            app.vy = vy;
            app.vz = vz;
        end

        %% func stft
        function TF = stft(app,f,wd)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % short-time Fourier transform with Guassian window
            %
            % Input:
            %      f - signal (real or complex)
            %      wd - std. deviation (sigma) of the Gaussian function
            % Output:
            %      TF - time-frequency distribution
            %
            % By V.C. Chen
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % f = app.f;
            % wd = app.wd;

            cntr = length(f)/2;
            sigma = wd;
            fsz = length(f);
            z = exp(-([1:fsz]-fsz/2).^2/(2*sigma^2))/(sqrt(2*pi)*sigma);
            for m=1:fsz
                mm = m-cntr+fsz/2;
                if (mm <= fsz & mm >= 1)
                    gwin(m) = z(mm);
                else
                    if (mm > fsz)
                        gwin(m) = z(rem(mm,fsz));
                    else
                        if(mm < 1)
                            mm1 = mm+fsz;
                            gwin(m) = z(mm1);
                        end
                    end
                end
            end
            winsz = length(gwin);
            x = zeros(1,fsz+winsz); % zero padding
            x(winsz/2+1:winsz/2+fsz) = f;
            for j = 1:fsz
                for k = 1:winsz
                    X(k) = gwin(k)*x(j+k);
                end
                TF(:,j) = fft(X).';
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end



        %% func rcsellipsoid
        function rcs = rcsellipsoid(app, a,b,c,phi,theta)

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % This program calculates the backscattered RCS of an ellipsoid
            % with semi-axis lengths of a, b, and c.
            % The source code is based on Radar Systems Analysis and Design Using
            % MATLAB, By B. Mahafza, Chapman & Hall/CRC 2000.
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            rcs = (pi*a^2*b^2*c^2)/(a^2*(sin(theta))^2*(cos(phi))^2+...
                b^2*(sin(theta))^2*(sin(phi))^2+c^2*(cos(theta))^2)^2;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end



        %% func ellipsoid2P

        function [X,Y,Z] = ellipsoid2P(app,P1,P2,a,b,c,N)

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % Construct an ellipsoid connecting two center points: P1 and P2.
            % Semi-axis lengths are a, b, and c.
            % N is the number of grid points for plotting the ellipsoid.
            %
            % By V.C. Chen
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            Cntr = (P1+P2)/2;  % ellipsoid center
            Lc = norm(P2-P1);

            % the axis defined by: P1+V*[0:Lc]
            V = (P1-P2)/Lc;   %normalized cylinder's axis-vector;
            U = rand(1,3);     %linear independent vector
            U = V-U/(U*V');    %orthogonal vector to V
            U = U/sqrt(U*U');  %orthonormal vector to V
            W = cross(V,U);    %vector orthonormal to V and U
            W = W/sqrt(W*W');  %orthonormal vector to V and U

            % generate the ellipsoid at (0,0,0)
            [Xc,Yc,Zc] = ellipsoid(0,0,0,a,b,c,N);

            A = kron(U',Xc);
            B = kron(W',Yc);
            C = kron(V',Zc);
            TMP = A+B+C;
            nt = size(TMP,2);

            X = TMP(1:nt,:)+Cntr(1);
            Y = TMP(nt+1:2*nt,:)+Cntr(2);
            Z = TMP(2*nt+1:end,:)+Cntr(3);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end


        %% calculate the position of bird's body
        function results = cal_body_position(app)

            app.dt = app.T/app.nt; % time interval
            app.ts = [0:app.dt:app.T-app.dt]; % time span
            app.f0 = app.c/app.lambda;
            app.nr = floor(2*sqrt(app.radarloc(1)^2+app.radarloc(2)^2+app.radarloc(3)^2)/app.rangeres);

            app.t = app.ts;
            app.ksi1 = app.A1*cos(2*pi*app.F0*app.t)+app.ksi10; % flapping angle

            % the elbow joint position in local coordinates
            app.x1 = app.V.*app.t;
            app.y1 = app.L1*cosd(app.ksi1);
            app.z1 = app.y1.*tand(app.ksi1);

            app.ksi2 = app.A2*cos(2*pi*app.F0*app.t)+app.ksi20;
            app.theta2 = app.C2*sin(2*pi*app.F0*app.t);
            % the wrist joint position in local coordinates
            app.d = app.theta2./cosd(app.ksi1-app.ksi2);
            app.y2 = app.L1*cosd(app.ksi1)+app.L2*cosd(app.theta2).*cosd(app.ksi1-app.ksi2);
            app.x2 = app.x1-(app.y2-app.y1).*tand(app.d);
            app.z2 = app.z1+(app.y2-app.y1).*tand(app.ksi1-app.ksi2);

            % bird body position
            app.Pb1(1,:) = -0.4+app.x1(:);
            app.Pb1(2,:) = zeros(size(app.x1));
            app.Pb1(3,:) = zeros(size(app.x1));
            app.Pb2(1,:) = 0.4+app.x1(:);
            app.Pb2(2,:) = zeros(size(app.x1));
            app.Pb2(3,:) = zeros(size(app.x1));

            % body center point
            app.Cb = (app.Pb1+app.Pb2)/2;


        end


        %%  calculate the radar data
        function results = cal_data(app)
            % translation
            % rotation first, then translation x1, y1, z1
            [app.x1, app.y1, app.z1] = app.rotate_and_translate(app.x1, app.y1, app.z1, app.vx, app.vy, app.vz, app.t);

            % rotation first, then translation x2, y2, z2
            [app.x2, app.y2, app.z2] = app.rotate_and_translate(app.x2, app.y2, app.z2, app.vx, app.vy, app.vz, app.t);

            % rotation first, then translation Pb1
            [app.Pb1(1,:), app.Pb1(2,:), app.Pb1(3,:)] = app.rotate_and_translate(app.Pb1(1,:), app.Pb1(2,:), app.Pb1(3,:), app.vx, app.vy, app.vz, app.t);

            % rotation first, then translation Pb2
            [app.Pb2(1,:), app.Pb2(2,:), app.Pb2(3,:)] = app.rotate_and_translate(app.Pb2(1,:), app.Pb2(2,:), app.Pb2(3,:), app.vx, app.vy, app.vz, app.t);

            % body center point
            app.Cb = (app.Pb1 + app.Pb2) / 2;

            % left upper arm position
            app.Pua11 = app.Cb;
            app.Pua21 = [app.x1; app.y1; app.z1];
            app.Puac1 = (app.Pua11+app.Pua21)/2; % upper arm center

            % left forearm position
            app.Pfa11 = app.Pua21;
            app.Pfa21 = [app.x2; app.y2; app.z2];
            app.Pfac1 = (app.Pfa11+app.Pfa21)/2; % forearm center

            % right upper arm position
            app.Pua12 = app.Cb;
            app.Pua22 = [app.x1; -app.y1; app.z1];
            app.Puac2 = (app.Pua12+app.Pua22)/2; % upper arm center

            % right forearm position
            app.Pfa12 = app.Pua22;
            app.Pfa22 = [app.x2; app.y2; app.z2];
            app.Pfac2 = (app.Pfa12+app.Pfa22)/2; % forearm center


            % radar returns
            % prepare data collection
            app.data = zeros(app.nr,app.nt);

            % radar returns from the body
            for k = 1:app.nt
                % distance from radar to bird body
                app.r_dist(:,k) = abs(app.Cb(:,k)-app.radarloc(:));
                app.distances(k) = sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2+app.r_dist(3,k).^2);
                % aspect vector of the body
                app.aspct(:,k) = app.Pb2(:,k)-app.Pb1(:,k);
                % calculate theta angle
                A = [app.radarloc(1)-app.Cb(1,k); app.radarloc(2)-app.Cb(2,k);...
                    app.radarloc(3)-app.Cb(3,k)];
                B = [app.aspct(1,k); app.aspct(2,k); app.aspct(3,k)];
                A_dot_B = dot(A,B,1);
                A_sum_sqrt = sqrt(sum(A.*A,1));
                B_sum_sqrt = sqrt(sum(B.*B,1));
                app.ThetaAngle(k) = acos(A_dot_B ./ (A_sum_sqrt .* B_sum_sqrt));
                app.PhiAngle(k) = asin((app.radarloc(2)-app.Cb(2,k))./...
                    sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2));
                app.rcs(k) = app.rcsellipsoid(app.Ba,app.Bb,app.Bc,app.PhiAngle(k),app.ThetaAngle(k));
                app.amp(k) = sqrt(app.rcs(k));
                PHs = app.amp(k)*(exp(-j*4*pi*app.distances(k)/app.lambda));
                app.data(floor(app.distances(k)/app.rangeres),k) = ...
                    app.data(floor(app.distances(k)/app.rangeres),k) + PHs;
            end

            % radar returns from the left upper arm wing
            for k = 1:app.nt
                % distance from radar to bird left upper arm
                app.r_dist(:,k) = abs(app.Puac1(:,k)-app.radarloc(:));
                app.distances(k) = sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2+app.r_dist(3,k).^2);
                % aspect vector of the body
                app.aspct(:,k) = app.Pua21(:,k)-app.Pua11(:,k);
                % calculate theta angle
                A = [app.radarloc(1)-app.Puac1(1,k); app.radarloc(2)-app.Puac1(2,k);...
                    app.radarloc(3)-app.Puac1(3,k)];
                B = [app.aspct(1,k); app.aspct(2,k); app.aspct(3,k)];
                A_dot_B = dot(A,B,1);
                A_sum_sqrt = sqrt(sum(A.*A,1));
                B_sum_sqrt = sqrt(sum(B.*B,1));
                app.ThetaAngle(k) = acos(A_dot_B ./ (A_sum_sqrt .* B_sum_sqrt));
                app.PhiAngle(k) = asin((app.radarloc(2)-app.Puac1(2,k))./...
                    sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2));
                app.rcs(k) = app.rcsellipsoid(app.Wa,app.Wb,app.Wc,app.PhiAngle(k),app.ThetaAngle(k));
                app.amp(k) = sqrt(app.rcs(k));
                PHs = app.amp(k)*(exp(-j*4*pi*app.distances(k)/app.lambda));
                app.data(floor(app.distances(k)/app.rangeres),k) = ...
                    app.data(floor(app.distances(k)/app.rangeres),k) + PHs;
            end

            % radar returns from the left forearm wing
            for k = 1:app.nt
                % distance from radar to bird body
                app.r_dist(:,k) = abs(app.Pfac1(:,k)-app.radarloc(:));
                app.distances(k) = sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2+app.r_dist(3,k).^2);
                % aspect vector of the body
                app.aspct(:,k) = app.Pfa21(:,k)-app.Pfa11(:,k);
                % calculate theta angle
                A = [app.radarloc(1)-app.Pfac1(1,k); app.radarloc(2)-app.Pfac1(2,k);...
                    app.radarloc(3)-app.Pfac1(3,k)];
                B = [app.aspct(1,k); app.aspct(2,k); app.aspct(3,k)];
                A_dot_B = dot(A,B,1);
                A_sum_sqrt = sqrt(sum(A.*A,1));
                B_sum_sqrt = sqrt(sum(B.*B,1));
                app.ThetaAngle(k) = acos(A_dot_B ./ (A_sum_sqrt .* B_sum_sqrt));
                app.PhiAngle(k) = asin((app.radarloc(2)-app.Pfac1(2,k))./...
                    sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2));
                app.rcs(k) = app.rcsellipsoid(app.Wa,app.Wb,app.Wc,app.PhiAngle(k),app.ThetaAngle(k));
                app.amp(k) = sqrt(app.rcs(k));
                PHs = app.amp(k)*(exp(-j*4*pi*app.distances(k)/app.lambda));
                app.data(floor(app.distances(k)/app.rangeres),k) = ...
                    app.data(floor(app.distances(k)/app.rangeres),k) + PHs;
            end

            % radar returns from the right upper arm wing
            for k = 1:app.nt
                % distance from radar to bird body
                app.r_dist(:,k) = abs(app.Puac2(:,k)-app.radarloc(:));
                app.distances(k) = sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2+app.r_dist(3,k).^2);
                % aspect vector of the body
                app.aspct(:,k) = app.Pua22(:,k)-app.Pua12(:,k);
                % calculate theta angle
                A = [app.radarloc(1)-app.Puac2(1,k); app.radarloc(2)-app.Pua21(2,k);...
                    app.radarloc(3)-app.Puac2(3,k)];
                B = [app.aspct(1,k); app.aspct(2,k); app.aspct(3,k)];
                A_dot_B = dot(A,B,1);
                A_sum_sqrt = sqrt(sum(A.*A,1));
                B_sum_sqrt = sqrt(sum(B.*B,1));
                app.ThetaAngle(k) = acos(A_dot_B ./ (A_sum_sqrt .* B_sum_sqrt));
                app.PhiAngle(k) = asin((app.radarloc(2)-app.Puac2(2,k))./...
                    sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2));
                app.rcs(k) = app.rcsellipsoid(app.Wa,app.Wb,app.Wc,app.PhiAngle(k),app.ThetaAngle(k));
                app.amp(k) = sqrt(app.rcs(k));
                PHs = app.amp(k)*(exp(-j*4*pi*app.distances(k)/app.lambda));
                app.data(floor(app.distances(k)/app.rangeres),k) = ...
                    app.data(floor(app.distances(k)/app.rangeres),k) + PHs;
            end

            % radar returns from the right forearm wing
            for k = 1:app.nt
                % distance from radar to bird body
                app.r_dist(:,k) = abs(app.Pfac2(:,k)-app.radarloc(:));
                app.distances(k) = sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2+app.r_dist(3,k).^2);
                % aspect vector of the body
                app.aspct(:,k) = app.Pfa22(:,k)-app.Pfa12(:,k);
                % calculate theta angle
                A = [app.radarloc(1)-app.Pfac2(1,k); app.radarloc(2)-app.Pfac2(2,k);...
                    app.radarloc(3)-app.Pfac2(3,k)];
                B = [app.aspct(1,k); app.aspct(2,k); app.aspct(3,k)];
                A_dot_B = dot(A,B,1);
                A_sum_sqrt = sqrt(sum(A.*A,1));
                B_sum_sqrt = sqrt(sum(B.*B,1));
                app.ThetaAngle(k) = acos(A_dot_B ./ (A_sum_sqrt .* B_sum_sqrt));
                app.PhiAngle(k) = asin((app.radarloc(2)-app.Pfac2(2,k))./...
                    sqrt(app.r_dist(1,k).^2+app.r_dist(2,k).^2));
                app.rcs(k) = app.rcsellipsoid(app.Wa,app.Wb,app.Wc,app.PhiAngle(k),app.ThetaAngle(k));
                app.amp(k) = sqrt(app.rcs(k));
                PHs = app.amp(k)*(exp(-j*4*pi*app.distances(k)/app.lambda));
                app.data(floor(app.distances(k)/app.rangeres),k) = ...
                    app.data(floor(app.distances(k)/app.rangeres),k) + PHs;
            end


        end



        function [x_new, y_new, z_new] = rotate_and_translate(app, x, y, z, vx, vy, vz, t)
            % define velocity
            v = [vx; vy; vz];

            if all(v == [0;0,;0])
                x_new = x ;
                y_new = y ;
                z_new = z ;
                return;
            end

            % unit vector
            v_unit = v / norm(v);

            % original direction(direction of x)
            initial_direction = [1; 0; 0];

            % rotation axis
            rotation_axis = cross(initial_direction, v_unit);
            rotation_axis = rotation_axis / norm(rotation_axis); % normalization

            % check the direction
            if all(v_unit == initial_direction)
                x_new = x + vx * t;
                y_new = y + vy * t;
                z_new = z + vz * t;
                return;
            end

            % compute rotate angle
            theta = acos(dot(initial_direction, v_unit));
            K = [0 -rotation_axis(3) rotation_axis(2);
                rotation_axis(3) 0 -rotation_axis(1);
                -rotation_axis(2) rotation_axis(1) 0];
            R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K);
            [m, n] = size(x);
            x_new = zeros(m, n);
            y_new = zeros(m, n);
            z_new = zeros(m, n);
            for ii = 1:m
                for jj = 1:n
                    new_position = R * [x(ii, jj); y(ii, jj); z(ii, jj)];
                    x_new(ii, jj) = new_position(1);
                    y_new(ii, jj) = new_position(2);
                    z_new(ii, jj) = new_position(3);
                end
            end
            x_new = x_new + vx * t;
            y_new = y_new + vy * t;
            z_new = z_new + vz * t;
        end


        function drawRadarTrackingaFlyingBird(app)
            app.cal_body_position()
            figure()
            for k = 1:200:app.nt
                clf
                hold on
                colormap([0.7 0.7 0.7])
                [x,y,z] = app.ellipsoid2P([-0.4+app.x1(k),0,0],[0.4+app.x1(k),0,0],0.1,0.1,0.4,30);
                [x, y, z] = app.rotate_and_translate(x, y, z, app.vx, app.vy, app.vz, app.t(k));
                surf(x,y,z) %
                % left wing
                [x,y,z] = app.ellipsoid2P([app.x1(k),0,0],[app.x1(k),app.y1(k),app.z1(k)],...
                    0.05,0.05,0.25,30);
                [x, y, z] = app.rotate_and_translate(x, y, z, app.vx, app.vy, app.vz, app.t(k));
                surf(x,y,z) %
                [x,y,z] = app.ellipsoid2P([app.x1(k),app.y1(k),app.z1(k)],[app.x2(k),app.y2(k),app.z2(k)],...
                    0.05,0.05,0.25,30);
                [x, y, z] = app.rotate_and_translate(x, y, z, app.vx, app.vy, app.vz, app.t(k));
                surf(x,y,z)
                % right wing
                [x,y,z] = app.ellipsoid2P([app.x1(k),0,0],[app.x1(k),-app.y1(k),app.z1(k)],...
                    0.05,0.05,0.25,30);
                [x, y, z] = app.rotate_and_translate(x, y, z, app.vx, app.vy, app.vz, app.t(k));
                surf(x,y,z) %
                [x,y,z] = app.ellipsoid2P([app.x1(k),-app.y1(k),app.z1(k)],[app.x2(k),-app.y2(k),app.z2(k)],...
                    0.05,0.05,0.25,30);
                [x, y, z] = app.rotate_and_translate(x, y, z, app.vx, app.vy, app.vz, app.t(k));
                surf(x,y,z)
                light
                lighting gouraud
                light('Position',[20 10 20],'Style','infinite');
                shading interp
                axis equal
                axis([-1,20,-5,5,-10,2])
                axis on
                grid on
                set(gcf,'Color',[1 1 1])
                view([30,15])
                % draw radar location
                plot3(app.radarloc(1),app.radarloc(2),app.radarloc(3),'-ro',...
                    'LineWidth',2,...
                    'MarkerEdgeColor','r',...
                    'MarkerFaceColor','y',...
                    'MarkerSize',10)
                % draw a line from radar to the target center
                app.Cb(1,k) = app.Cb(1,k)+app.vx * app.t(k);
                app.Cb(2,k) = app.Cb(2,k)+app.vy * app.t(k);
                app.Cb(3,k) = app.Cb(3,k)+app.vz * app.t(k);
                line([app.radarloc(1) app.Cb(1,k)],[app.radarloc(2) app.Cb(2,k)],...
                    [app.radarloc(3) app.Cb(3,k)],...
                    'color',[0.4 0.7 0.7],'LineWidth',1.5,'LineStyle','-.')
                xlabel('X')
                ylabel('Y')
                zlabel('Z')
                title('Radar Tracking a Flying Bird')
                drawnow

            end

        end


        % Button pushed function: drawRangeProfilesButton
        function drawRangeProfiles(app)


            app.cal_body_position()
            app.cal_data()


            figure()
            colormap(jet(256))
            imagesc([1,app.nt],[0,app.nr*app.rangeres],20*log10(abs(app.data)+eps))
            xlabel('Pulses')
            ylabel('Range (m)')
            title('Range Profiles of Bird Flapping Wings')
            axis xy
            clim = get(gca,'CLim');
            set(gca,'CLim',clim(2) + [-20 10]);
            colorbar
            drawnow

        end

        % Button pushed function: drawmicroDopplersignatureButton
        function drawmicroDopplersignature(app)
            app.cal_body_position()


            app.cal_data()


            % micro-Doppler signature
            x = sum(app.data);
            np = app.nt;

            dT = app.T/length(app.ts);
            F = 1/dT;
            dF = 1/app.T;

            wd = 512;
            wdd2 = wd/2;
            wdd8 = wd/8;
            ns = np/wd;

            % calculate time-frequency micro-Doppler signature
            %disp('Calculating segments of TF distribution ...')
            for k = 1:ns

                sig(1:wd,1) = x(1,(k-1)*wd+1:(k-1)*wd+wd);
                TMP = app.stft(sig,16);
                TF2(:,(k-1)*wdd8+1:(k-1)*wdd8+wdd8) = TMP(:,1:8:wd);
            end
            TF = TF2;
            %disp('Calculating shifted segments of TF distribution ...')
            TF1 = zeros(size(TF));
            for k = 1:ns-1

                sig(1:wd,1) = x(1,(k-1)*wd+1+wdd2:(k-1)*wd+wd+wdd2);
                TMP = app.stft(sig,16);
                TF1(:,(k-1)*wdd8+1:(k-1)*wdd8+wdd8) = TMP(:,1:8:wd);
            end
            %disp('Removing edge effects ...')
            for k = 1:ns-1
                TF(:,k*wdd8-8:k*wdd8+8) = ...
                    TF1(:,(k-1)*wdd8+wdd8/2-8:(k-1)*wdd8+wdd8/2+8);
            end

            % display final time-frequency signature
            figure()
            colormap(jet(256))
            imagesc([0,app.T],[-F/2000,F/2000],20*log10(fftshift(abs(TF),1)+eps))
            xlabel('Time (s)')
            ylabel('Doppler (kHz)')
            %title('Micro-Doppler Signature of Flapping Wings')
            axis xy
            clim = get(gca,'CLim');
            set(gca,'CLim',clim(2) + [-60 0]);
            %    set(gca,'FontSize',24)
            colorbar
            drawnow


        end


        function save_square_spectrogram(app, img_path)
            app.cal_body_position()


            app.cal_data()


            % micro-Doppler signature
            x = sum(app.data);
            np = app.nt;

            dT = app.T/length(app.ts);
            F = 1/dT;
            dF = 1/app.T;

            wd = 512;
            wdd2 = wd/2;
            wdd8 = wd/8;
            ns = np/wd;

            % calculate time-frequency micro-Doppler signature
            %disp('Calculating segments of TF distribution ...')
            for k = 1:ns

                sig(1:wd,1) = x(1,(k-1)*wd+1:(k-1)*wd+wd);
                TMP = app.stft(sig,16);
                TF2(:,(k-1)*wdd8+1:(k-1)*wdd8+wdd8) = TMP(:,1:8:wd);
            end
            TF = TF2;
            %disp('Calculating shifted segments of TF distribution ...')
            TF1 = zeros(size(TF));
            for k = 1:ns-1

                sig(1:wd,1) = x(1,(k-1)*wd+1+wdd2:(k-1)*wd+wd+wdd2);
                TMP = app.stft(sig,16);
                TF1(:,(k-1)*wdd8+1:(k-1)*wdd8+wdd8) = TMP(:,1:8:wd);
            end
            %disp('Removing edge effects ...')
            for k = 1:ns-1
                TF(:,k*wdd8-8:k*wdd8+8) = ...
                    TF1(:,(k-1)*wdd8+wdd8/2-8:(k-1)*wdd8+wdd8/2+8);
            end

            % display final time-frequency signature
            fig = figure('Visible', 'off');
            colormap(jet(256))
            imagesc([0,app.T],[-F/2000,F/2000],20*log10(fftshift(abs(TF),1)+eps))
            %title('Micro-Doppler Signature of Flapping Wings')
            axis xy
            clim = get(gca,'CLim');
            set(gca,'CLim',clim(2) + [-60 0]);
            %    set(gca,'FontSize',24)
            drawnow
            set(gca, 'Position', [0 0 1 1]);
            set(gcf,'color','white')
            set(gcf,'units','pixels','position',[0 0 512 512])
            set(gcf,'WindowState','maximized')
            frame=getframe(gca);
            img = frame2im(frame);
            imwrite(img,img_path);
            close(fig);
        end

    end



end







