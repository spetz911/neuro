%%% 4_1
%~ Использовать вероятностную нейронную сеть для классификации точек в случае,
%~ когда классы не являются линейно разделимыми. Проверить качество обучения.
clear;
clc;

%% 1.1
%~ Обучающее множество, соответствующее своему варианту, занести в отчет.
%~ Отобразить входное множество и эталонное распределение по классам.
%~ Для отображения необходимо номер класса −1 заменить на 0.
P = [0.1 0.6; 0 1.4; -0.9 -1.2; -1.2 -1.4; -0.3 -0.2; -0.5 0.7; 1.2 0.9; -1 1.4; 0.5 -1.1; 0 -1.3; 1.3 0; 1 -0.1]';
T = [-1 -1 1 1 1 1 -1 1 1 1 -1 1];
T(T == 1) = 2;
T(T == -1) = 1;
vT = ind2vec(T);

%% 1.2
%~ Построить вектор индексов классов с помощью функции ind2vec.
%~ Перед этим в векторе, содержащим распределение по классам заменить −1 на 1, а 1 заменить на 2.
%~ Константу SPREAD задать равной 0.3. Создать сеть с помощью функции newpnn. Отобразить структуру сети.
net = newpnn(P, vT, 0.3);
% net = train(net, P, vT);
% display(net);

%% 1.3
%~ После создания сети занести в отчет весовые коэффициенты и смещения для двух слоев.

net_IW = net.IW{1,1}
net_b = net.b{1}
net_LV = net.LW{2,1}



%% 1.4
%~ Проверить качество обучения: случайным образом задать 5 точек и классифицировать их.
%~ Если результаты неудовлетворительные, то изменить значение SPREAD и создать новую сеть.
%~ Отобразить сетку, дополнительные точки, обучающую выборку, и результаты распределения по классам.
%~ Результаты занести в отчет. Перед отображением преобразовать выход сети с помощью функции vec2ind,
%~ а также преобразовать коды классов для отображения.
randP = rands(2, 5);
randT = vec2ind(sim(net, randP));
randT(randT == 1) = 0;
randT(randT == 2) = 1;
T(T == 1) = 0;
T(T == 2) = 1;

net_IW = net.IW{1,1}
net_b = net.b{1}
net_LV = net.LW{2,1}
net_b = net.b{2}


figure
hold on
xlabel('Input Vector P');
ylabel('Target Vector T');
plotpv(horzcat(P, randP), horzcat(T, randT)), grid
title('Training Vectors');
hold off

waitforbuttonpress
quit

