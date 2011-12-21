%% 1
clear;
clc;
% ������� ������
P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [1 1 0 0 0 1]; % ������������� ����� �� �������

net = newp([-5 5; -5 5], [0, 1]); % ������ ����� ����������� ����������

%% 1.3.1
% �������������� ���� ���������� ����������
net.inputweights{1,1}.initFcn = 'rands'; % ����� ������� ������������� ����� ��������
net.biases{1}.initFcn = 'rands'; % ����� ������� �������� ����� ��������
net = init(net); % ������������������ ����

IW = net.IW{1,1};
b = net.b{1};
display(net)
%% 1.3.2
% �������� �������� �� ������� �����������
%> ���� �������� ������� � �����
passes = 2;
for j = 1:passes;
    for i = 1:6
        p = P(:,i);
        t = T(:,i);
        a = sim(net, p);
        e = t - a;
        IW = net.IW{1,1};
        b = net.b{1};
        IW = IW + e*p';
        b = b + e;
        net.IW{1,1} = IW;
        net.b{1} = b;
    end
end

Y = sim(net, P);
E = T - Y;
perf = mae(E);

%% 1.3.3
plotpv(P, T), grid
plotpc(net.IW{1,1}, net.b{1})

%% 1.4.1
% ���������������� ���� ���������� ����������
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

%% 1.4.2
net.trainParam.epochs = 50;
[net, tr] = train(net, P, T);

%% 1.4.3 
% ��������� �������� �������� �� 3� ��������� ������
rP = rands(2, 3);
rT = sim(net, rP);

plotpv(horzcat(P, rP), horzcat(T, rT)), grid % ������������� ������� �������������
plotpc(net.IW{1,1}, net.b{1})

%% 2.1 
% �������� �������, ����� ������ ����� ������� �������������
clear;
clc;
P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [1 0 0 0 1 1]; % ������������� ����� �� �������

%% 2.2
% ���������������� ��������
net = newp([-5 5; -5 5], [0, 1]); % ������ ����� ����������� ����������
net.inputweights{1,1}.initFcn = 'rands'; % ������ ������� ������������� ����� ��������
net.biases{1}.initFcn = 'rands'; % ������ ������� �������� ����� ��������
net = init(net); % �������������������� ����

%% 2.3
% ������� ���� � ���������� ����������
net.trainParam.epochs = 50;
[net, tr] = train(net, P, T);

plotpv(P, T), grid
plotpc(net.IW{1,1}, net.b{1})
clear;
clc;
%% 3.1
P = [
    2.6 3.6 0.1 0.8 -3.1 2.4; 
    -3.4 4.8 3.8 -3.5 -1 3.2
];

T = [
    1 1 0 0 0 1;
    0 1 1 0 0 1
];

%% 3.2
% ������� ����
net = newp([-5 5; -5 5], [0 1; 0 1]); % ������� ����� ����������

%% 3.3
% ���������������� ���� ���������� ����������
net.inputweights{1,1}.initFcn = 'rands'; % ������ ������� ������������� ����� ��������
net.biases{1}.initFcn = 'rands'; % ������ ������� �������� ����� ��������
net = init(net);

%% 3.4
% ������� ���� � ���������� ����������
net.trainParam.epochs = 50;
[net, tr] = train(net, P, T);

net.IW{1,1};
net.b{1};

%% 3.5
% ��������� �������� �������� �� 5� ��������� ������
rP = rands(2, 5);
rT = sim(net, rP);

plotpv(horzcat(P, rP), horzcat(T, rT)), grid % ������������� ������� �������������
plotpc(net.IW{1,1}, net.b{1})