% Analisi dati da simulazioni
clear
clc
close all

frames_m = load('/Users/annat/Dati/AIM simulation/AIM-simulation/2L90frames_m.mat','frames_m');
frames_m1 = frames_m.frames_m;

frames_m = load('/Users/annat/Dati/AIM simulation/AIM-simulation/4L90frames_m.mat','frames_m');
frames_m2 = frames_m.frames_m;

size(frames_m) 

%%
figure(1)
for t = 1000:size(frames_m1,1)
    subplot(1,2,1)
    tmp = frames_m1(t,:,:);
    mat2d = squeeze(tmp);
    imagesc(mat2d); colorbar
    title(sprintf('Fase liquida - Frame %d', t))
    %clim([-260 260])
    subplot(1,2,2)
    tmp = frames_m2(t,:,:);
    mat2d = squeeze(tmp);
    imagesc(mat2d); colorbar
    title(sprintf('Asters - Frame %d', t))
    sgtitle('Taglia 90x90, T = 0.2')
    %clim([-260 260])
    pause(0.01)
end
%%
figure(1)
for t = 1:size(frames_m1,1)
    tmp = frames_m1(t,:,:);
    mat2d = squeeze(tmp);
    imagesc(mat2d); colorbar
    title(sprintf('Frame %d', t))
    %clim([-15 15])
    pause(0.01)
end

%%

figure(2)
for t = 1:size(frames_m,1)
    tmp = frames_m(t,:,:);
    mat2d = squeeze(tmp);
    profilo_magnet = sum(sum(mat2d));
    disp(profilo_magnet)
    %plot(t,profilo_magnet, '*')
    %ylim([-1,1])
    pause(0) 
end


%%
figure(3)
plot(order_param(1,:))


%%
close all
clear
clc

path1 = 'T0.500asymmetry_frames_m_track.mat';
path2 = 'T0.500asymmetry_track.mat';
path3 = 'T0.050asymmetry_frames_m_track.mat';
path4 = 'T0.050asymmetry_track.mat';

frames_m = load(path1,'frames_m');
track = load(path2,'track');

frames_m = frames_m.frames_m;
track = track.track;
dim = size(track);

figure(1)
for t = 1:size(frames_m,1)
    hold on
    tmp = frames_m(t,:,:);
    mat2d = squeeze(tmp);
    imagesc(mat2d); colorbar
    colormap gray
    title(sprintf('Frame %d', t))
    
    for i = 1:dim(1)
        if track(i,t,3) == 1
           if track(i,t,1) == 0
              track(i,t,1) = 1;
           end
           if track(i,t,2) == 0
              track(i,t,2) = 1;
           end
           plot(track(i,t,1), track(i,t,2), 'Color', 'r', 'Marker','>', 'MarkerSize', 10, 'LineWidth', 2)
        else
           plot(track(i,t,1), track(i,t,2), 'Color', 'b', 'Marker','<', 'MarkerSize', 10, 'LineWidth', 2)
        end
    end
    pause(0.0005)
    ylim([1,101])
    xlim([1,201])
    hold off
end

T = 0.05;
D = 1;
dt = 1/(4*D + exp(1/T));
p = D*dt;