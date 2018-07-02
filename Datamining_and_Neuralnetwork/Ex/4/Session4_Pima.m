sDbupa = som_read_data('pima.txt');
sDbupa=som_normalize(sDbupa, 'var');
sMap=som_make(sDbupa, 'algorithm', 'batch', 'msize', [13, 7], ...
'lattice', 'hexa', 'training', 'default'); 
% choose algortihm, grid size, type of grid, training time 
%(arguments given are default agruments)
sMap=som_autolabel(sMap, sDbupa, 'vote');
figure;
som_show(sMap,'norm','d');   % basic visualization
figure;
som_show(sMap,'umat','all','empty','Labels');    % UMatrix

som_show_add('label',sMap,'Textsize',8,'TextColor','r','Subplot',2);
% labels

%SORT DATASET according to labels with e.g. spreadsheet program (is done)
h1 = som_hits(sMap,sDbupa.data(str2num(cell2mat(sDbupa.labels))==0,:)); % ? 0’s (not diabetes)
h2 = som_hits(sMap,sDbupa.data(str2num(cell2mat(sDbupa.labels))==1,:)); % ? 1’s (diabetes)
pause
figure;
colormap('gray');
som_show(sMap,'umat','all','empty','Labels');    % UMatrix
som_show_add('label',sMap,'Textsize',8,'TextColor','r','Subplot',2);  
% labels
som_show_add('hit',[h1, h2],'MarkerColor',[1 0 0; 0 1 0],'Subplot',1)
% Hit diagram, normal liver = red, liver disorder = green.
[q,t] = som_quality(sMap,sDbupa)