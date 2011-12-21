%% 1.1
clear;
clc;
P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [1 1 0 0 0 1];


%% 1.2
net = newlind(P,T);

%% 1.3
IW = net.IW{1,1};
b = net.b{1};

display(IW);
display(b);
plotpv(P,T),grid;
plotpc(net.IW{1},net.b{1});

%% 2.1
clear;
clc;
h = 0.01;
t0 = 0;
t1 = 2.5;
size = (t1-t0)/h + 1;
points = zeros(1,size);
x = t0:h:t1;
for i = 1:size
    t = (i-1)*h;
    points(i) = sin(0.5*t*t - 5*t);
end
P = con2seq(points);
T = P;

%% 2.2
net = newlin([-1,1],1,0,0.1);
net.inputweights{1,1}.delays = [1 2 3 4 5];
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);
IW = net.IW{1,1};
b = net.b{1};
display(IW);
display(b);

%% 2.4
net.adaptParam.passes = 130;
[net, y, E, pf, af] = adapt(net, P(6:size), P(6:size), P(1:5));
display(sqrt(mse(E)));
view(net);

%% 2.5
figure
hold on
plot(x(6:size), cell2mat(y), '-r');
plot(x, cell2mat(T), '-b');
hold off
grid on
legend('output', 'etalon');
title('Graphics');

%% 2.6
figure
hold on
plot(x(6:size), cell2mat(E), '-r');
hold off
grid on
legend('error');
title('Error');

%% 3.1
clear;
clc;
h = 0.01;
t0 = 0;
t1 = 2.5;
 
size = (t1-t0)/h + 1;
points = zeros(1,size);
pointsT = zeros(1,size);
x = t0:h:t1;
for i = 1:size
    t = (i-1)*h;
    points(i) = sin(-3*t*t+5*t+10);
    pointsT(i) = sin(-3*t*t + 5*t - 3)/3;
end
P = con2seq(points);
T = con2seq(pointsT);

%% 3.2
net = newlin([-1,1],1,0,0.1);
net.inputweights{1,1}.delays = [1 2 3 4 5];
 
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);
IW = net.IW{1,1};
b = net.b{1};
display(IW);
display(b);

%% 3.3
net.adaptParam.passes = 70;
[net, y, E, pf, af] = adapt(net, P(6:size), T(6:size), P(1:5));
display(sqrt(mse(E)));

%% 3.4
figure
hold on
plot(x(6:size), cell2mat(P(6:size)), '-r');
plot(x(6:size), cell2mat(y), '-g');
plot(x(6:size), cell2mat(T(6:size)), '-b');
hold off
grid on
legend('input','output', 'etalon');
title('Graphics');

%% 3.5
figure
hold on
plot(x(6:size), cell2mat(E), '-r');
hold off
grid on
legend('error');
title('Error');