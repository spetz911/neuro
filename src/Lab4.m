%% 1.1
clear;
clc;
P = [0.1 0.6; 0 1.4; -0.9 -1.2; -1.2 -1.4; -0.3 -0.2; -0.5 0.7; 1.2 0.9; -1 1.4; 0.5 -1.1; 0 -1.3; 1.3 0; 1 -0.1]';
T = [-1 -1 1 1 1 1 -1 1 1 1 -1 1];
T(T == 1) = 2; 
T(T == -1) = 1; 
vT = ind2vec(T);
%% 1.2
net = newpnn(P, vT, 0.3);
display(net);
%% 1.3
net = init(net);
%% 1.4
rP = rands(2, 5);
rvT = sim(net, P);
rT = vec2ind(rvT);
rT(rT == 1) = 0; 
rT(rT == 2) = 1; 
T(T == 1) = 0; 
T(T == 2) = 1; 
%% 1.5
plotpv(horzcat(P, rP), horzcat(T, rT)), grid
%% 2.1
clear;
clc;
P = 0:0.025:6;
T = sin(P.^2-2*P+3);
%% 2.2
[trainInd, valInd, testInd] = dividerand(size(P, 2),0.8,0.0,0.2);
P1 = P(trainInd);
T1 = T(trainInd);
%% 2.3
net = newrb(P1, T1, 1e-8, 0.25, size(P1, 2), 10);
display(net);
%% 2.4
result = sim(net, P);
error = T - result;
sqrt(mse(error))
%% 2.5
plot(P, result, 'b', P, T, 'r'), grid;
legend('Network output', 'Target');
title('Radial basis network');

%%
%2.6
plot(P, error), grid;
legend('Error');
%% 3.1
clear;
clc;
P = 0:0.025:6;
T = sin(P.^2-2*P+3);
%% 3.2
[trainInd, valInd, testInd] = dividerand(size(P, 2),0.8,0.0,0.2);
P1 = P(trainInd);
T1 = T(trainInd);
%% 3.3
net = newgrnn(P1, T1, 0.05);
display(net);
%% 3.4
result = sim(net, P);
error = T - result;
sqrt(mse(error))
%% 3.5
plot(P, result, 'b', P, T, 'r'), grid;
legend('Network output', 'Target');
title('Generalized regression neural network');

%% 3.6
plot(P, error), grid;
legend('Error');

