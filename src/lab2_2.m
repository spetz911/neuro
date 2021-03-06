%%% 2_2
%~ Построить и обучить линейную сеть с задержками,
%~ которая будет аппроксимировать первую функцию из варианта задания.
clear;
clc;

%% 2.1
%~ Построить обучающее множество:
%~ в качестве входного множества использовать значения первого входного сигнала на заданном интервале;
%~ преобразовать входное множество к последовательности входных образцов с помощью функции con2seq;
%~ эталонные выходы сети должны совпадать с входным множеством.
t0 = 0;
h = 0.01;
t1 = 2.5;
size = (t1-t0)/h + 1;
values = zeros(1,size);

args = t0:h:t1;
for i = 1:size
    t = args(i);
    values(i) = sin(0.5*t*t - 5*t);
end

P = con2seq(values);

%% 2.2
%~ Задать задержки от 1 до 5. Задать скорость обучения равной 0.1.
net = newlin(P, values, [1 2 3 4 5], 0.1);
% net.inputweights{1,1}.delays = [1 2 3 4 5];

%% 2.3
%~ Инициализировать сеть случайными значениями.
%~ Занести в отчет весовые коэффициенты и смещения.
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

net_IW = net.IW{1,1}
net_b = net.b{1}


%% 2.4
%~ Выполнить адаптацию с числом циклов равным 50.
%~ Занести в отчет величину ошибки обучения с помощью функций sqrt(mse(e)).
%~ Поскольку сеть имеет задержки, то в функцию адаптации необходимо отдельно передать
%~ первые 5 элементов входной последовательности для инициализации задержек (входной параметр Pi).
%~ В противном случае задержки будут инициализированы нулями,
%~ что приведет к увеличению ошибки обучения при выполнении адаптации.
%~ В дальнейшем использовать входную и выходную последовательности, начиная с 6 элемента.
net.adaptParam.passes = 50;
[net, y, E, pf, af] = adapt(net, P(6:size), P(6:size), P(1:5));
sqrt_mse = sqrt(mse(E))
% view(net);

%% 2.5
%~ Отобразить на графике эталонные значения и предсказанные сетью,
%~ также отобразить точки заданного интервала.
%~ С помощью функции legend подписать кривые.
%~ Перед отображением привести значения из массива ячеек к матричному виду с помощью функции cell2mat.
figure('Name', 'Network results')
hold on
plot(args(6:size), cell2mat(y), '-r');
plot(args, values, '-b');
hold off
grid on
legend('output', 'original');


% waitforbuttonpress

%% 2.6
%~ Отобразить ошибку обучения. На графике отобразить сетку и точки заданного интервала.
figure('Name', 'Error')
hold on
plot(args(6:size), cell2mat(E), '-r');
hold off
grid on
legend('error');

waitforbuttonpress
quit

