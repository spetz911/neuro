%%% 5_3
%~ Построить и обучить нелинейную авторегрессионную сеть для идентификации динамической системы.
%~ Предсказать с помощью обученной сети значения функции.
%~ Горизонт прогноза составляет 20 временных отсчетов.
clear;
clc;

%% 3.1
%~ Построить обучающее множество.
%~ Входная последовательность формируется из входного управляющего сигнала u(k) и выходного сигнала y(k).
%~ u(k) = f (k),
%~ k ∈ [0, 10] c шагом h = 0.01
%~ y(k + 1) = y(k) / (1 + (k − 1)*y^2)  + u^3(k)
%~ Функция f (k) определяется вариантом задания.
%~ Последовательность целевых выходов задает выходной сигнал y(k).
%~ Преобразовать обучающее множество с помощью функции con2seq.
%~ Для обучения сети использовать обучающее множество без 20 последних элементов.
h = 0.01;
k = 0:h:10;
u = sin(-2*k.^2+7*k);
[~, n] = size(u);

y = zeros(1, n);
y(1) = u(1)^3;
y(2) = u(2)^3;
for k = 2:n-1
    y(k+1) = y(k)/(1+y(k-1)^2) +  u(k)^3;
end

P = con2seq(u(1:end-20));
T = con2seq(y(1:end-20));

%% 3.2
%~ Создать NARX сеть с последовательно-параллельной архитектурой с помощью функции narxnet.
%~ Задать задержки от 1 до 5 для каждого из входов сети. Число нейронов скрытого слоя задать равным 10.
%~ Отобразить структуру сети c помощью функций display.
net = narxnet(1:5, 1:5, 10);
display(net);
view(net);


%% 3.3
%~ С помощью функции preparets сформировать массивы ячеек для функции обучения,
%~ содержащие обучающее множество и значения для инициализации задержек входного слоя (P,T,Pi соответственно).
%~ Если при выполнении заданий используется версия MATLAB, которая не поддерживает эту функцию,
%~ то обучать и выполнять расчет выходов сети без инициализации задержек.
[Ps,Pi,Ai,Ts] = preparets(net,P,{},T);


%% 3.4
%~ Задать параметры обучения: число эпох обучения (net.trainParam.epochs) равным 600,
%~ предельное значение критерия обучения (net.trainParam.goal) равным 10^-8.
net.trainParam.epochs = 600;
net.trainParam.goal = 1e-8;


%% 3.5
%~ Произвести обучение сети с помощью метода, заданного по умолчанию.
%~ Если необходимо, то произвести обучение несколько раз.
%~ Если результаты неудовлетворительные или наблюдается переобучение,
%~ то изменить число нейронов или величину задержек.
%~ Занести в отчет весовые коэффициенты и смещения для двух слоев после обучения.
%~ Занести в отчет графики Performance, Training State, а также окно Neural Network Training.
net = train(net, Ps, Ts, Pi, Ai);
net = train(net, Ps, Ts, Pi, Ai);


%% 3.6
%~ Занести в отчет величину ошибки обучения с помощью функций sqrt(mse(e)),
%~ где e задает разность между выходными значениями сети и эталонными значениями обучающего множества.
%~ При расчете выхода сети инициализировать линии задержки входов сети.
out = sim(net, Ps, Pi, Ai, Ts);
err = cell2mat(out) - cell2mat(Ts);
sqrt_mse = sqrt(mse(err))

IW = net.IW{1}
LW = net.LW{2,1}
b1 = net.b{1}
b2 = net.b{2}

%% 3.7
%~ Отобразить на графике эталонные значения и предсказанные сетью,
%~ также отобразить входной управляющий сигнал и точки заданного интервала.
%~ С помощью функции legend подписать кривые.
figure;
title('etalon and predicted values1');
xlabel('k');
ylabel('y');
hold on;
plot(cell2mat(Ts), 'b'), grid;
plot(cell2mat(out), 'r');
hold off;
legend('output', 'etalon');


%% 3.8
%~ Отобразить ошибку обучения. На графике отобразить сетку и точки заданного интервала.
figure;
title('error1');
xlabel('k');
ylabel('error');
hold on;
plot(err, 'b'), grid;
hold off;
legend('error');


%% 3.9
%~ Выполнить с помощью сети прогноз на оставшиеся 20 временных отсчетов.
%~ Для этого инициализировать линию задержки последними 5 значениями из обучающего множества.
P25 = con2seq(u(end-25:end));
T25 = con2seq(u(end-25:end));
[Ps,Pi,Ai,Ts] = preparets(net,P25,{},T25);
out = sim(net, Ps, Pi);


%% 3.10
%~ Занести в отчет величину ошибки прогнозирования с помощью функций sqrt(mse(e)),
%~ где e задает разность между значениями, предсказанными сетью,
%~ и последними 20 эталонными значениями обучающего множества.
err = cell2mat(out) - cell2mat(Ts);
sqrt_mse = sqrt(mse(err))

%% 3.11
%~ Отобразить на графике эталонные значения и предсказанные сетью,
%~ также отобразить точки заданного интервала.
%~ С помощью функции legend подписать кривые.
figure;
title('etalon and predicted values2');
xlabel('k');
ylabel('y');
hold on;
plot(cell2mat(out), '.-r'),grid;
plot(cell2mat(Ts), 'b');
hold off;
legend('output', 'etalon');

%% 3.12
%~ Отобразить ошибку прогнозирования. На графике отобразить сетку и точки заданного интервала.
figure;
title('error2');
xlabel('k');
ylabel('error');
hold on;
plot(err, 'b'), grid;
legend('error');
hold off;

waitforbuttonpress
quit


