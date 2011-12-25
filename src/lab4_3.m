%%% 4_3
%~ Построить обобщенно-регрессионную нейронную сеть, которая будет выполнять
%~ аппроксимацию функции из лабораторной работы No 3.
clear;
clc;


%% 3.1
%~ Построить обучающее множество. Обучающим множеством является таблица значений:
%~ точки из заданного интервала и значения функции в них.
P = 0:0.025:6;
T = sin(P.^2-2*P+3);


%% 3.2
%~ Создать сеть с помощью функции newgrnn(P1,T1,SPREAD).
%~ Константу SPREAD задать равной 2 ∗ h, где h — величина шага для заданной функции.
%~ Произвести разделение обучающей выборки на обучающее и тестовое подмножества
%~ с помощью функции(dividerand) в соотношении 80% и 20%.
%~ Индексы обучающего подмножества использовать для создания сети.
%~ P 1 = P (trainInd);
%~ T 1 = T (trainInd);

[trainInd, valInd, testInd] = dividerand(size(P, 2),0.8,0.0,0.2);
P1 = P(trainInd);
T1 = T(trainInd);
net = newgrnn(P1, T1, 0.05);


%% 3.3
%~ Отобразить структуру сети.
display(net);


%% 3.4
%~ Если результаты неудовлетворительные, то изменить значение SPREAD и создать новую сеть.


%% 3.5
%~ Занести в отчет величину ошибки обучения с помощью функций sqrt(mse(e)).
%~ Для расчета ошибки обучения необходимо получить выходы сети для всего входного множества:
%~ error = T - net(P).
result = sim(net, P);
error = T - result;
sqrt(mse(error))


%% 3.6
%~ Отобразить на графике эталонные значения и предсказанные сетью, также отобразить точки заданного интервала.
%~ С помощью функции legend подписать кривые. Также указать полное название метода обучения.
plot(P, result, 'b', P, T, 'r'), grid;
legend('Network output', 'Target');
title('Generalized regression neural network');


%% 3.7
%~ Отобразить ошибку обучения. На графике отобразить сетку и указать шкалу времени.
plot(P, error), grid;
legend('Error');


