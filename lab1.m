%% 1
clear;
clc;
% Входные образы
P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [1 1 0 0 0 1]; % распределение точек по классам

net = newp([-5 5; -5 5], [0, 1]); % создаём новый однослойный перцептрон

%% 1.3.1
% Инициализируем сеть случайными значениями
net.inputweights{1,1}.initFcn = 'rands'; % задаём функцию инициализации весов рандомом
net.biases{1}.initFcn = 'rands'; % задаём функцию смещений весов рандомом
net = init(net); % переинициализируем сеть

IW = net.IW{1,1};
b = net.b{1};
display(net)
%% 1.3.2
% алгоритм обучения по правилу Розенблатта
%> весь алгоритм занести в отчёт
passes = 2;
for j = 1:passes;
    for i = 1:6
        p = P(:,i);
        t = T(:,i);
        a = sim(net, p);
        e = t - a;
        IW = net.IW{1,1};
        b = net.b{1};
        IW = IW + e*p';
        b = b + e;
        net.IW{1,1} = IW;
        net.b{1} = b;
    end
end

Y = sim(net, P);
E = T - Y;
perf = mae(E);

%% 1.3.3
plotpv(P, T), grid
plotpc(net.IW{1,1}, net.b{1})

%% 1.4.1
% Инициализировать сеть случайными значениями
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

%% 1.4.2
net.trainParam.epochs = 50;
[net, tr] = train(net, P, T);

%% 1.4.3 
% Проверить качество обучения на 3х случайных точках
rP = rands(2, 3);
rT = sim(net, rP);

plotpv(horzcat(P, rP), horzcat(T, rT)), grid % конкатенируем матрицы горизонтально
plotpc(net.IW{1,1}, net.b{1})

%% 2.1 
% Изменить выборку, чтобы классы стали линейно нераздилимыми
clear;
clc;
P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [1 0 0 0 1 1]; % распределение точек по классам

%% 2.2
% Инициализировать рандомом
net = newp([-5 5; -5 5], [0, 1]); % создаём новый однослойный перцептрон
net.inputweights{1,1}.initFcn = 'rands'; % задать функцию инициализации весов рандомом
net.biases{1}.initFcn = 'rands'; % задать функцию смещений весов рандомом
net = init(net); % переинициализировать сеть

%% 2.3
% Обучить сеть и отобразить результаты
net.trainParam.epochs = 50;
[net, tr] = train(net, P, T);

plotpv(P, T), grid
plotpc(net.IW{1,1}, net.b{1})
clear;
clc;
%% 3.1
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
net = newp([-5 5; -5 5], [0 1; 0 1]); % создать новый перцептрон

%% 3.3
% Инициализировать сеть случайными значениями
net.inputweights{1,1}.initFcn = 'rands'; % задать функцию инициализации весов рандомом
net.biases{1}.initFcn = 'rands'; % задать функцию смещений весов рандомом
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