%% 1.1 Обучающее множество занести в отчет.
clear;
clc;
P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [1 1 0 0 0 1];


%% 1.2 Создать сеть с помощью функции newlind.
net = newlind(P,T);

%% 1.3 Занести в отчет весовые коэффициенты и смещения. Отобразить сетку, обучающую
%      выборку, и дискриминантную линию. Результаты занести в отчет.
IW = net.IW{1,1};
b = net.b{1};

display(IW);
display(b);

plotpv(P,T),grid;
plotpc(net.IW{1},net.b{1});
waitforbuttonpress

%% 1.4 Создать сеть с помощью функции newlin. Задать максимальную скорость обучения
%      с помощью функции maxlinlr. Сконфигурировать сеть под обучающее множество и отоб-
%      разить структуру сети.
net = newlin(P,T, [0], maxlinlr(P));
display(net)

%% 1.5 Инициализировать сеть случайными значениями. Занести в отчет весовые коэффи-
%      циенты и смещения.
net.inputweights{1,1}.initFcn = 'rands'; % функция инициализации весов
net.biases{1}.initFcn = 'rands'; % функция смещений весов
net = init(net); % инициализация

display 'inputweights && biases:'
net.IW{1,1}
net.b{1}


% 1.6 Провести обучение сети с помощью функции train c числом эпох равным 100. Занести
%     в отчет окно Neural Network Training, график Performance . Занести в отчет весовые
%     коэффициенты и смещения.
net.trainParam.epochs = 100;
display 'training... please wait'
[net, tr] = train(net, P, T);

net.IW{1,1}
net.b{1}

figure
hold on
plotpv(P, T), grid
plotpc(net.IW{1,1}, net.b{1})
hold off

waitforbuttonpress



quit

