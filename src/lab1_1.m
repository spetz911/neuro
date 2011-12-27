%%% 1_1
clear;
clc;

%% 1.1
P = [
	2.6 3.6 0.1 0.8 -3.1 2.4; 
	-3.4 4.8 3.8 -3.5 -1 3.2
];
n = 6;

% распределение точек по классам
T = [1 1 0 0 0 1];

%% 1.2
% новый однослойный перцептрон
net = newp([-5 5; -5 5], [0, 1]);

%% 1.3.1
% Инициализируем случайными значениями
net.inputweights{1,1}.initFcn = 'rands'; % функция инициализации весов
net.biases{1}.initFcn = 'rands'; % функция смещений весов
net = init(net);

% Отобразить структуру сети
display(net)

IW = net.IW{1,1}
b = net.b{1}

%% 1.3.2
%~ Рассчитать два цикла обучения сети по правилу Розенблатта.
%~ Для расчета выходов сети использовать функцию sim.
%~ В качестве показателя качества обучения использовать функцию mae.
%~ Занести в отчет весовые коэффициенты и смещения после расчета каждой эпохи (итерации).

display('study...')
rounds = 2;
for k = 1:rounds;
	for i = 1:n
		p = P(:,i);
		t = T(:,i);
		a = sim(net, p);
		e = t - a;
		net.IW{1,1} = net.IW{1,1} + e*p';
		net.b{1} = net.b{1} + e;
	end
end

IW = net.IW{1,1}
b = net.b{1}

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
randP = rands(2, 3);
randT = sim(net, rP);

% конкатенируем матрицы горизонтально
plotpv(horzcat(P, randP), horzcat(T, randT)), grid
plotpc(net.IW{1,1}, net.b{1})

waitforbuttonpress
quit
