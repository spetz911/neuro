clear;
clc;
P = 0:0.01:2.2;
T = sin(0.5*P.^2-5*P);
net = timedelaynet(1:5,8);
display(net);
n = size(T, 2);
T1 = T(1:(n-20));
sT1 = con2seq(T1);

net = timedelaynet(1:5, 8);
display(net);

[Xs, Xi, Ai, Ts] = preparets(net,sT1, sT1);
net.trainParam.epochs = 600;
net.trainParam.goal = 1e-8;
net = train(net, Xs, Ts, Xi, Ai);

out = sim(net, Xs, Xi, Ai, Ts);
err = cell2mat(out) - cell2mat(Ts);
display(sqrt(mse(err)));

%1.7 графики
figure;
title('predicated and etalon values');
xlabel('t');
ylabel('y');
hold on;
plot(P(6:n-20), cell2mat(out), '.-r'), grid;
plot(P(6:n-20), cell2mat(Ts), 'b');
hold off;
legend('output', 'etalon');

%1.8 ошибка
figure;
title('error');
xlabel('t');
ylabel('y');
hold on;
plot(P(6:n-20), err, 'b'), grid;
hold off;
legend('error');

%1.9 оставшиеся 20 временных отсчетов
P25 = con2seq(T(n-20-5+1:end));
T25 = P25;
[Ps,Pi,Ai,Ts] = preparets(net,P25,T25);
out = sim(net, Ps, Pi);

%1.10 ошибка прогнозирования
err = cell2mat(out) - cell2mat(T25(5+1:end));
display(sqrt(mse(err)));

%1.11
figure;
title('etalon and predicated values');
xlabel('t');
ylabel('y');
hold on;
plot(P(n-20+1:end), cell2mat(out), '.-r'),grid;
plot(P(n-20+1:end), cell2mat(Ts), 'b');
hold off;
legend('predicted fcn', 'etalon');

%1.12
figure;
title('error');
xlabel('t');
ylabel('y');
hold on;
plot(P(n-20+1:end), err, 'b'),grid;
hold off;
legend('error');

%2
clear;
clc;
k1 = 0:0.01:1;
n1 = size(k1, 2);
t1 = -ones(1, n1);
p1 = sin(4*pi*k1);
k2 = 0.92:0.01:4.07;
n2 = size(k2, 2);
t2 = ones(1, n2);
p2 = cos(-2*k2.^2 + 7*k2);

R = [2 4 7];

P = [repmat(p1,1,R(1)), p2, repmat(p1,1,R(2)), p2, repmat(p1,1,R(3)), p2];
T = [repmat(t1,1,R(1)), t2, repmat(t1,1,R(2)), t2, repmat(t1,1,R(3)), t2];

Pseq = con2seq(P);
Tseq = con2seq(T);

d1 = 0:4;
d2 = 0:4;
net = distdelaynet({d1, d2}, 8, 'trainbr');

display(net);

[Ps,Pi,Ai,Ts] = preparets(net,Pseq,Tseq);
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-5;
net = train(net, Ps, Ts, Pi, Ai);

out = sim(net, Ps, Pi, Ai, Ts);
err = cell2mat(out) - cell2mat(Ts);
display(sqrt(mse(err)));

figure;
title('etalon and predicated values');
xlabel('t');
ylabel('y');
hold on;
plot(cell2mat(out), 'b'), grid;
plot(cell2mat(Ts), 'g');
hold off;
legend('output', 'etalon');

figure;
title('error');
xlabel('t');
ylabel('y');
hold on;
plot(err, 'b'),grid;
hold off;
legend('error');

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
legend('output', 'etalon');

%2.11 ошибка
figure;
xlabel('t');
ylabel('y');
hold on;
plot(err, 'b'), grid;
legend('error');

clear;
clc;
%3.1 построить обучающее множество
h = 0.01;
k = 0:h:10;
width = 20;
u = sin(-2*k.^2+7*k);
[~, n] = size(u);
display(n);
y = zeros(1, n);
y(1) = u(1)^3;
y(2) = u(2)^3;
for k = 2:n-1
    y(k+1) = y(k)/(1+y(k-1)^2) +  u(k)^3;
end

Pseq = con2seq(u(1:n-width));
Tseq = con2seq(y(1:n-width));

d = 5;
net = narxnet(1:d, 1:d, 10);
display(net);
view(net);

[Ps,Pi,Ai,Ts] = preparets(net,Pseq,{},Tseq);

net.trainParam.epochs = 600;
net.trainParam.goal = 1e-8;

net = train(net, Ps, Ts, Pi, Ai);

%3.6 ошибка
out = sim(net, Ps, Pi, Ai, Ts);
err = cell2mat(out) - cell2mat(Ts);
display(sqrt(mse(err)));

%3.7
figure;
title('etalon and predicated values');
xlabel('k');
ylabel('y');
hold on;
plot(cell2mat(out), '.-r'), grid;
plot(cell2mat(Ts), 'b');
hold off;
legend('output', 'etalon');

%3.8
figure;
title('error');
xlabel('k');
ylabel('error');
hold on;
plot(err, 'b'),grid;
hold off;
legend('error');

%3.9
P25 = con2seq(u(n-width-d+1:end));
T25 = con2seq(u(n-width-d+1:end));
[Ps,Pi,Ai,Ts] = preparets(net,P25,{},T25);
out = sim(net, Ps, Pi);

%3.10
err = cell2mat(out) - cell2mat(Ts);
display(sqrt(mse(err)));

%3.11
figure;
title('etalon and predicated values');
xlabel('k');
ylabel('y');
hold on;
plot(cell2mat(out), '.-r'),grid;
plot(cell2mat(Ts), 'b');
hold off;
legend('output', 'etalon');

%3.12
figure;
title('error');
xlabel('k');
ylabel('error');
hold on;
plot(err, 'b'), grid;
hold off;
legend('error');
