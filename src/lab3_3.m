%%% 3_3
%~ Построить и обучить двухслойную нейронную сеть прямого распространения, ко-
%~ торая будет выполнять аппроксимацию функции. Для обучения использовать алгоритм,
%~ реализующий метод оптимизации функций многих переменных второго порядка. Функция
%~ и метод обучения определяются вариантом задания.
%~ Последовательности шагов для выполнения 2 и 3 этапов работы совпадают.
clear;
clc;

%% 3.1
%~ Построить обучающее множество. Обучающим множеством является таблица
%~ значений: точки из заданного интервала и значения функции в них.
P = 0:0.025:6;
T = sin(P.^2-2*P+3);

%% 3.2
%~ Создать сеть с помощью функции feedforwardnet. Сконфигурировать сеть под обу-
%~ чающее множество с помощью функции configure. Число нейронов скрытого слоя задать
%~ равным 10. Использовать активационные функции, заданные по умолчанию. Алгоритм обу-
%~ чения определяется вариантом задания. Разделение обучающей выборки на обучающее,
%~ контрольное, и тестовое подмножества производится с помощью функции заданной по
%~ умолчанию (dividerand) в соотношении 60%, 20%, 20%. Отобразить структуру сети.
net = feedforwardnet(18);
net = configure(net, P, T);
net.trainFcn = 'trainlm';
net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0.2;
net.divideFcn
net.divideParam
net.divideMode
display(net);

%% 3.3
%~ Инициализировать сеть с помощью функции, заданной по умолчанию.
%~ Занести в отчет весовые коэффициенты и смещения для двух слоев.
net = init(net);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}

%% 3.4
%~ Задать параметры обучения: значения параметров для некоторых методов обучения
%~ описаны выше, число эпох обучения (net.trainParam.epochs) равно 600, предельное
%~ значение критерия обучения (net.trainParam.goal) равно 10−8 .
net.trainParam.epochs = 600;
net.trainParam.goal = 1e-8;


%% 3.5
%~ Выполнить обучение сети с помощью функции train. Если необходимо, то произвести обучение несколько раз.
%~ Если результаты неудовлетворительные или наблюдается переобучение, то изменить число нейронов в функции feedforwardnet,
%~ увеличить число эпох обучения или уменьшить предельное значение критерия обучения.
%~ Занести в отчет весовые коэффициенты и смещения для двух слоев. Занести в отчет графики Performance, Training State,
%~ а также окно Neural Network Training, если это возможно для данного метода обучения.
net = train(net, P, T);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}

%% 3.6
%~ Занести в отчет величину ошибки обучения с помощью функций sqrt(mse(e)).
%~ Поскольку обучающая выборка разделяется на несколько частей, то для получения ошибки
%~ обучения необходимо получить выходы сети для всего входного множества:
%~ error = T - net(P).
result = sim(net, P);
error = T - result;
sqrt(mse(error))

%% 3.7
%~ Отобразить на графике эталонные значения и предсказанные сетью, также отоб-
%~ разить точки заданного интервала. С помощью функции legend подписать кривые. Также
%~ указать полное название метода обучения.
plot(P, result, 'b', P, T, 'r'),grid;
legend('Network output', 'Target');
title('Levenberg-Marquardt backpropagation');

%% 3.8
%~ Отобразить ошибку обучения. На графике отобразить сетку и указать шкалу времени.
plot(P, error),grid;
legend('Error');

