%% 1.1
clear;
clc;

P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];
n = 6;

% распределение точек по классам
T = [1 1 0 0 0 1];

% новый однослойный перцептрон
net = newp([-5 5; -5 5], [0, 1]);

%% 1.3.1
% Инициализируем случайными значениями
net.inputweights{1,1}.initFcn = 'rands'; % функция инициализации весов
net.biases{1}.initFcn = 'rands'; % функция смещений весов
net = init(net);

% Отобразить структуру сети
display(net)

IW = net.IW{1,1};
b = net.b{1};

%% 1.3.2
% алгоритм обучения по правилу Розенблатта
iterations = 2;
for j = 1:iterations;
    for i = 1:n
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
display(perf)

%% 1.3.3
plotpv(P, T), grid
plotpc(net.IW{1,1}, net.b{1})

waitforbuttonpress

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

waitforbuttonpress
quit
