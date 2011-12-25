%% 1.1
clear;
clc;
P = [0.1 0.6; 0 1.4; -0.9 -1.2; -1.2 -1.4; -0.3 -0.2; -0.5 0.7; 1.2 0.9; -1 1.4; 0.5 -1.1; 0 -1.3; 1.3 0; 1 -0.1]';

T = [-1 -1 1 1 1 1 -1 1 1 1 -1 1];
T(T == -1) = 0; 

%% 1.2
net = feedforwardnet(5, 'trainrp');
net = configure(net, P, T);
net.divideFcn='';
display(net);

%% 1.3
net = init(net);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}

%% 1.4
net.trainParam.epochs = 1200;
net.trainParam.goal = 1e-5;
net = train(net, P, T);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}

%% 1.5
rP = rands(2, 5);
rT = sim(net, rP);
rT(rT < 0) = 0; 
rT(rT >= 0) = 1; 
plotpv(horzcat(P, rP), horzcat(T, rT)), grid

%% 2.1
clear;
clc;
P = 0:0.025:6;
T = sin(P.^2-2*P+3);
%% 2.2
net = feedforwardnet(14);
net = configure(net, P, T);
net.trainFcn = 'traincgb';
net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0.2;
net.divideFcn
net.divideParam
net.divideMode
display(net);
%% 2.3
net = init(net);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}
%% 2.4
net.trainParam.epochs = 600;
net.trainParam.goal = 1e-8;
net = train(net, P, T);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}
%% 2.5
result = sim(net, P);
error = T - result;
sqrt(mse(error))
%% 2.6
plot(P, result, 'b', P, T, 'r'),grid;
legend('Network output', 'Target');
title('Conjugate gradient backpropagation with Powell-Beale restarts');
%% 2.7
plot(P, error),grid;
legend('Error');

%% 3.1
clear;
clc;
P = 0:0.025:6;
T = sin(P.^2-2*P+3);
%% 3.2
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
net = init(net);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}

%% 3.4
net.trainParam.epochs = 600;
net.trainParam.goal = 1e-8;
net = train(net, P, T);
net.IW{1,1}
net.b{1}
net.LW{2,1}
net.b{2}
%% 3.5
result = sim(net, P);
error = T - result;
sqrt(mse(error))
%% 3.6
plot(P, result, 'b', P, T, 'r'),grid;
legend('Network output', 'Target');
title('Levenberg-Marquardt backpropagation');
%% 3.7
plot(P, error),grid;
legend('Error');
