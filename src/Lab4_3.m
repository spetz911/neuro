clear;
clc;
P = 0:0.025:6;
T = sin(P.^2-2*P+3);

[trainInd, valInd, testInd] = dividerand(size(P, 2),0.8,0.0,0.2);
P1 = P(trainInd);
T1 = T(trainInd);

net = newgrnn(P1, T1, 0.05);
display(net);

result = sim(net, P);
error = T - result;
sqrt(mse(error))

plot(P, result, 'b', P, T, 'r'), grid;
legend('Network output', 'Target');
title('Generalized regression neural network');

%%
%2.8
plot(P, error), grid;
legend('Error');




