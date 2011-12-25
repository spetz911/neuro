clear;
clc;
%2.1 последовательность обучающего множества
h = 0.01;
k1 = 0:h:1;
[~,n1] = size(k1);
t1 = -ones(1, n1);
p1 = sin(4*pi*k1);
k2 = 0.92:h:4.07;
[~,n2] = size(k2);
t2 = ones(1, n2);
p2 = sin(-2*k2.^2 + 7*k2);

R = [2 4 7];

P = [repmat(p1,1,R(1)), p2, repmat(p1,1,R(2)), p2, repmat(p1,1,R(3)), p2];
T = [repmat(t1,1,R(1)), t2, repmat(t1,1,R(2)), t2, repmat(t1,1,R(3)), t2];

Pseq = con2seq(P);
Tseq = con2seq(T);

%2.2 создание сети
d = 1:5;
net = layrecnet(d, 14);
display(net);
view(net);
%2.3 формирование массивов
[Ps,Pi,Ai,Ts] = preparets(net,Pseq,Tseq);

%2.4 параметры обучения
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-5;

%2.5 обучение
net = train(net, Ps, Ts, Pi, Ai);
%load net6_2.mat
%load net6_2lm.mat

%2.6 
out = sim(net, Ps, Pi, Ai, Ts);
err = cell2mat(out) - cell2mat(Ts);
display(sqrt(mse(err)));

%2.7 предсказанные и эталонные значения
figure;
title('etalon and predicated values');
xlabel('t');
ylabel('y');
hold on;
plot(cell2mat(out), 'b'), grid;
plot(cell2mat(Ts), 'g');
hold off;

%2.8 ошибка
figure;
title('error');
xlabel('t');
ylabel('y');
hold on;
plot(err, 'b'), grid;
hold off;

%2.9 
R = [1 6 6];

P = [repmat(p1,1,R(1)), p2, repmat(p1,1,R(2)), p2, repmat(p1,1,R(3)), p2];
T = [repmat(t1,1,R(1)), t2, repmat(t1,1,R(2)), t2, repmat(t1,1,R(3)), t2];
Pseq = con2seq(P);
Tseq = con2seq(T);

[Ps,Pi,Ai,Ts] = preparets(net,Pseq,Tseq);

%2.10
out = sim(net, Ps, Pi, Ai, Ts);
err = cell2mat(out) - cell2mat(Ts);
display(sqrt(mse(err)));

%2.10 предсказанные и эталонные значения
figure;
title('etalon and predicated values');
xlabel('t');
ylabel('y');
hold on;
plot(cell2mat(out), 'b'), grid;
plot(cell2mat(Ts), 'g');
hold off;

%2.11 ошибка
figure;
title('error');
xlabel('t');
ylabel('y');
hold on;
plot(err, 'b'), grid;
hold off;