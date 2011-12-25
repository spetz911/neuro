%% 3.1
clear;
clc;

P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [
    1 1 0 0 0 1;
    0 1 1 0 0 1
];

%% 3.2
% Создать сеть
net = newp([-5 5; -5 5], [0 1; 0 1]);

%% 3.3
% Инициализировать сеть случайными значениями
net.inputweights{1,1}.initFcn = 'rands'; % функция инициализации весов
net.biases{1}.initFcn = 'rands'; % функция смещений весов
net = init(net);

%% 3.4
% Обучить сеть и отобразить результаты
net.trainParam.epochs = 50;
[net, tr] = train(net, P, T);

net.IW{1,1};
net.b{1};

%% 3.5
% Проверить качество обучения на 5и случайных точках
rP = rands(2, 5);
rT = sim(net, rP);

plotpv(horzcat(P, rP), horzcat(T, rT)), grid % конкатенируем матрицы горизонтально
plotpc(net.IW{1,1}, net.b{1})
waitforbuttonpress
quit


