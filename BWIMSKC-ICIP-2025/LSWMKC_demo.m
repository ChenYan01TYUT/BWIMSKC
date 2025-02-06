clear
clc
warning off;

path = './';
addpath(genpath(path));
%%
global neibour
neibour = 5;    %initial neighbours
pk=0;
qk=3;

dataName = 'binaryalphadigs';
disp(dataName);
load([path,'datasets/',dataName,'_v2'],'X','Y');%加载数据文件，其中KH是多个核矩阵（每个视图对应一个核矩阵），Y是标签（类别信息）。
Y(Y==-1)=2;%将Y中标签为-1的部分替换为2。
numclass = length(unique(Y)); %获取聚类数，即标签的种类数量。
numNeighbors = 5;  % 每个样本的邻居数

[nSmp, mSmp] = size(X);
options = [];
options.NeighborMode = 'KNN';
options.k = 5;%近邻数为5
options.WeightMode = 'HeatKernel';
A = constructW(X,options);

DSsym = 1./sqrt(max(sum(A, 2), eps));
Gnorm = (DSsym * DSsym') .* A; % 直接利用稀疏性
Gnorm = (Gnorm + Gnorm') / 2;
Gnorm = sparse(Gnorm);
L = speye(nSmp) - Gnorm; % 使用稀疏单位矩阵

KH_all = zeros(nSmp, nSmp, 4);  % 初始化为4轮循环

L_half = L / 2;
I_L_half = speye(nSmp, nSmp) - L_half;

% 提前计算L_p的初始值
L_p = speye(nSmp, nSmp);  % L_half^0 = I，初始化为单位矩阵

% 预先计算L_p的所有幂
L_p_precomputed = zeros(nSmp, nSmp, 4);
L_p_precomputed(:,:,1) = L_p;  % L_p^0（即单位矩阵）
for pk_val = 1:3
    L_p_precomputed(:,:,pk_val+1) = L_p_precomputed(:,:,pk_val) * L_half;  % 递推计算L_p的幂
end

% 预计算L_q（q = 0, 1, 2, 3）
L_q_precomputed = zeros(nSmp, nSmp, 4); 
L_q_precomputed(:,:,1) = speye(nSmp, nSmp);  % (I - L/2)^0 = I
L_q_precomputed(:,:,2) = I_L_half;  % (I - L/2)^1 = I - L/2

for q_val = 2:3
    L_q_precomputed(:,:,q_val+1) = L_q_precomputed(:,:,q_val) * I_L_half;  % 利用上一轮结果递推计算
end

for i = 1:4
    % 获取预计算的L_q
    L_q = L_q_precomputed(:,:,qk+1);  % 获取当前qk对应的L_q
    % 获取预计算的L_p
    L_p = L_p_precomputed(:,:,pk+1);  % 获取L_p的第pk次幂
    % 计算Beta函数
    B_value = beta(pk+1, qk+1);  % 计算Beta函数
    G = (L_p * L_q) / (2 * B_value);
    X_G = G*X;
    % 存储KH_all的结果
    KH_all(:, :, i) = (G * X_G) * (X_G' * G');

    % 更新pk和qk
    pk = pk + 1;
    qk = qk - 1;
end
%L_half = L / 2; % 预计算L的一半
%I = speye(nSmp, nSmp);
%for i = 1:4
%    L_p = L_half^(pk); % 计算L的幂次
%    L_q = (I - L_half)^(qk); % 计算L_q
%    G = (L_p * L_q) / (2 * beta(pk+1, qk+1)); 
%    X = G * X;
%    KH_all(:, :, i) = (G * X) * (X' * G');
%    pk = pk + 1;
%    qk = qk - 1;
%end

%%
alpha_range = 2.^[0:1:10];%设置一个参数alpha的取值范围，这里是2^0到2^10的值，表示在一个指数范围内变化。

for alpha_indx = 1:length(alpha_range)%对每个alpha值（从alpha_range中选取）进行循环。
    alpha = alpha_range(alpha_indx);%调用Graph_main函数，它接收核矩阵KH和alpha作为输入，返回图谱Kstar、簇标签Z、gamma（可能是正则化参数）、omega（可能是某种权重）、obj（目标函数值）。
    
    [Kstar,Z,gamma,omega,obj] = Graph_main(KH_all,alpha);
    
    Kstar = kcenter(Kstar);%对Kstar进行中心化和归一化处理。
    Kstar = knorm(Kstar);
    [H, ~] = eigs(Kstar, numclass, 'la');%对Kstar进行特征值分解，选择最大的numclass个特征值及其对应的特征向量。H是特征向量矩阵。
    [res(:,alpha_indx)] = myNMIACCV2(H,Y,numclass);%调用myNMIACCV2函数计算评估指标，包括ACC（准确率）、NMI（归一化互信息）、Pur（纯度）、Rand（Rand指数）。H是特征向量，Y是真实标签，numclass是类别数。
    
    fprintf('ACC:%4.4f \t NMI:%4.4f \t Pur:%4.4f \t Rand:%4.4f \n',...
        [ res(1,alpha_indx),res(2,alpha_indx),res(3,alpha_indx),res(4,alpha_indx) ]);%输出评估指标。
end
%%
[~,max_indx] = max(res(1,:,:),[],'all','linear'); %找到准确率（ACC）最大值对应的索引。
res_opt = res(:,max_indx)%选择最优结果，res_opt包含与最大准确率对应的评估指标。

fprintf('ACC_max:%4.4f \t NMI_max:%4.4f \t Pur_max:%4.4f \t Rand_max:%4.4f \t indx:%4.0f \n',...
    [ res(1,max_indx) res(2,max_indx) res(3,max_indx) res(4,max_indx) max_indx]);%输出最优的评估指标（ACC、NMI、Pur、Rand）及其对应的索引。

% figure;
% plot(obj, 'LineWidth', 2, 'Color', [0, 0.5, 1], 'Marker', 'o', 'MarkerFaceColor', 'r');  % 设置线条的宽度、颜色和标记
% plot(res(1,:), 'LineWidth', 2, 'Color', [1 0 0], 'MarkerFaceColor', [1 0 0])
% grid on;  % 添加网格
% xlabel('Iteration number', 'FontSize', 12, 'FontWeight', 'bold');  % 设置x轴标签
% ylabel('Objective Function Value', 'FontSize', 12, 'FontWeight', 'bold');  % 设置y轴标签
% title('Convergence Plot', 'FontSize', 14, 'FontWeight', 'bold');  % 设置标题
% set(gca, 'FontSize', 12);  % 设置坐标轴字体大小
% set(gca, 'LineWidth', 1.5);  % 设置坐标轴线宽

% figure;

% yyaxis left  % 使用左侧y轴显示目标函数
% plot( obj, 'o-', 'LineWidth', 2, 'Color', [0 0 1], 'MarkerFaceColor', [0 0 1]);
% xlabel('Iteration');
% ylabel('Objective Value');
% hold on;

% yyaxis right  % 使用右侧y轴显示性能指标
% plot( res(1,:), 's-', 'LineWidth', 2, 'Color', [1 0 0], 'MarkerFaceColor', [1 0 0]);
% plot( res(2,:), '^-', 'LineWidth', 2, 'Color', [0 1 0], 'MarkerFaceColor', [0 1 0]);
% plot( res(3,:), 'd-', 'LineWidth', 2, 'Color', [1 0 1], 'MarkerFaceColor', [1 0 1]);
% % 添加图例
% ylim([0 1]);
% legend('Loss', 'ACC', 'NMI', 'Purity', 'RI', 'Location', 'best');

% 格式化图表
% grid on;
figure;
plot(1:20, obj(1:20), 'o-', 'LineWidth', 2, 'Color', [0 0.447 0.741], 'MarkerFaceColor', [0 0.447 0.741], 'MarkerSize', 8);
xlabel('Iteration', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Objective Value', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 16, 'LineWidth', 1.5); % 设置坐标轴字体大小和线宽
xlim([1 20]); % 设置 x 轴范围为 1 到 20
legend('Objective Value', 'Location', 'northeast'); % 添加图例
hold off;
