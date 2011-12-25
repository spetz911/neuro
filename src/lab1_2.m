%% 2.1
% Изменить выборку, чтобы классы стали линейно нераздилимыми
clear;
clc;

P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

% распределение точек по классам
T = [1 0 0 0 1 1];

%% 2.2
% новый однослойный перцептрон
net = newp([-5 5; -5 5], [0, 1]);
% Инициализируем случайными значениями
net.inputweights{1,1}.initFcn = 'rands'; % функция инициализации весов
net.biases{1}.initFcn = 'rands'; % функция смещений весов
net = init(net);

%% 2.3
% Обучить сеть и отобразить результаты
net.trainParam.epochs = 50;
display 'training... please wait'
[net, tr] = train(net, P, T);

plotpv(P, T), grid
plotpc(net.IW{1,1}, net.b{1})
waitforbuttonpress
quit
