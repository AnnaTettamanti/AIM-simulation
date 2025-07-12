% Analisi dati da simulazioni & creazione video
clear; clc; close all;

% Carica i dati
d1 = load('/Users/annat/Dati/AIM simulation/AIM-simulation/2L90frames_m.mat','frames_m');
frames_m1 = d1.frames_m;
d2 = load('/Users/annat/Dati/AIM simulation/AIM-simulation/5L90frames_m.mat','frames_m');
frames_m2 = d2.frames_m;

% Scegli i limiti di colore comuni (calcolali sui due dataset)
all_data = [frames_m1(:); frames_m2(:)];
cmin = prctile(all_data,0.1);  % scarta i valori estremi
cmax = prctile(all_data,99.9);

% Prepara il video writer
v = VideoWriter('AIM_simulation.mp4','MPEG-4');
v.FrameRate = 10;        % 10 fps
v.Quality   = 75;        % qualità media per file più leggero
open(v);

% Crea la figura con layout fisso
hFig = figure('Units','pixels','Position',[100 100 600 300]);
set(hFig,'Color','w');

for t = 1000:size(frames_m1,1)
    % Subplot 1
    ax1 = subplot(1,2,1);
    mat2d = squeeze(frames_m1(t,:,:));
    imagesc(mat2d); 
    axis square off;          % rende quadrato e toglie assi
    title(sprintf('Fase liquida – Frame %d', t),'FontSize',10)

    % Subplot 2
    ax2 = subplot(1,2,2);
    mat2d = squeeze(frames_m2(t,:,:));
    imagesc(mat2d);
    axis square off;
    title(sprintf('Asters – Frame %d', t),'FontSize',10)

    % Titolo generale
    sgtitle('Taglia 90×90, T = 0.2','FontSize',12)

    % Cattura il frame
    frame = getframe(hFig);
    writeVideo(v,frame);
end

close(v);
disp('Video salvato come AIM_simulation.mp4');
