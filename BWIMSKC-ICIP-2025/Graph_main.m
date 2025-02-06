function [Kstar, Z, gamma, omega, obj] = Graph_main(KH,alpha)%该函数接受两个输入参数：KH 是一个包含多个核矩阵的三维数组（每个核对应一个视图），alpha 是一个超参数，用于控制不同核的重要性。
%输出包括：Kstar（更新后的图谱矩阵），Z（每个数据点的簇标签），gamma（可能是正则化系数），omega（核权重向量），以及 obj（目标函数的值）。
global neibour%声明 neibour 为全局变量，用于定义每个节点的邻居数。

iter = 0;
MaxIter = 100;
flag = 1;%初始化迭代计数器 iter 为 0，最大迭代次数 MaxIter 设置为 100，flag 用于控制是否继续迭代。

[num,num,numker] = size(KH); %获取输入的核矩阵 KH 的尺寸。numker 是核矩阵的数量，num 是样本数量（假设所有核矩阵的大小相同）。

omega = sqrt(ones(numker,1)/numker); %初始化核权重向量，每个核的权重初始为相同，且其值的平方和为 1。
avgKer = mycombFun(KH,omega);%调用 mycombFun 函数计算加权平均核矩阵。omega 是各个核的权重，KH 是多个核矩阵。
Kstarold = avgKer;             %初始图谱 Kstar 为核矩阵的加权平均值。
alpha_initial = 0;             %初始化超参数 alpha 为 0。
[Z_old, gamma] = update_gamma(KH,neibour,omega,Kstarold,alpha_initial); 
Z_initial = Z_old;%调用 update_gamma 函数更新节点标签 Z 和正则化参数 gamma。Z_old 存储初始的簇标签，gamma 是更新后的正则化系数。

Kstar_fnorm = [];
term_1 = [];
term_2 = [];
term_3 = [];
obj = [];%初始化一些变量来存储在迭代过程中计算的目标函数的各个项（term_1, term_2, term_3），以及最终的目标值 obj。

while flag
    iter = iter+1;
    %--------update weight---------------
    omega = zeros(1,numker);
    delta = zeros(1,numker);%初始化 omega 为零向量，delta 为零向量。
    for p =1: numker
        delta(1,p) = trace(KH(:,:,p)*Z_old);%对于每个核 p，计算核矩阵 KH(:,:,p) 与簇标签 Z_old 的迹（即矩阵的对角元素和），并将结果存储在 delta 向量中。
    end
    omega = delta./(norm(delta,2));%最后通过 delta 的 L2 范数归一化，更新核的权重 omega。
    
    %--------update Z--------------
    [Z] = update_Z(Kstarold,KH,alpha,gamma,omega); 
    Z_old = Z;%调用 update_Z 函数根据当前的图谱矩阵 Kstarold，核矩阵 KH，参数 alpha，正则化参数 gamma 和核权重 omega 更新簇标签 Z。
    
    %--------update K*---------------
    [Kstar] = update_Kstar(Z);%调用 update_Kstar 函数根据当前的簇标签 Z 更新图谱矩阵 Kstar。
    
    %--------calculate K* norm-------
    Kstar_fnorm(end+1) = norm(Kstar-Kstarold,'fro');%计算当前图谱矩阵 Kstar 和上一次的图谱矩阵 Kstarold 之间的 Frobenius 范数，并将其存储在 Kstar_fnorm 中。
    Kstarold = Kstar;%更新 Kstarold 为当前的 Kstar，以便在下一次迭代中使用。
    
    %--------covergence--------------
    if (iter>=20 && abs((Kstar_fnorm(iter)-Kstar_fnorm(iter-1)))/norm(Kstar,'fro')<1e-3) || iter>MaxIter
        flag =0;
    end
    %检查收敛条件：如果图谱矩阵的变化足够小（即相邻两次迭代的 Frobenius 范数变化小于 1e-3），或者迭代次数超过最大迭代次数，则停止迭代。
    
    %-----------obj------------------
    term_1_temp = 0;
    for p = 1:numker
        term_1_temp = term_1_temp + (- trace(omega(p) * KH(:,:,p) * Z));
    end
    term_1(end+1) = term_1_temp;%计算目标函数的第一项 term_1，它是核矩阵加权后与簇标签 Z 的迹的负值。
    
    for i = 1:num
        term_2_temp(i) = norm(Z(i,:),2)^2;
    end
    term_2(end+1) = gamma'*term_2_temp';%计算目标函数的第二项 term_2，它是簇标签 Z 每一行的 L2 范数的平方和。
    
    term_3(end+1) = alpha.*norm(Kstar - Z, 'fro')^2;%计算目标函数的第三项 term_3，它是图谱矩阵与簇标签矩阵之间的 Frobenius 范数的平方，乘以 alpha。
    
end
obj = term_1+term_2+term_3;%目标函数 obj 是上述三项的总和，它表示当前迭代中的优化目标。
