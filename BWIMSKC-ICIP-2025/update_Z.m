function [Z] = update_Z(Kstar,KH,alpha,gamma,omega)%该函数接收五个输入参数：Kstar：当前的图谱矩阵，表示节点之间的相似度。KH：包含多个核的三维矩阵，每个核表示节点间的不同关系。alpha：正则化参数，用于控制图谱矩阵 Kstar 的影响。gamma：每个节点的正则化参数，用于调整每个节点的簇标签。omega：每个核的权重，用于加权不同的核。

[num,num,numker] = size(KH);%获取核矩阵 KH 的尺寸

for i = 1: num  %迭代遍历每个节点 i，对每个节点的簇标签进行更新。
    Kv_temp = zeros(1,num);%Kv_temp 用于存储节点 i 与所有其他节点的加权距离。它是所有核矩阵加权后得到的距离总和。
    for p = 1:numker
        Kv_temp = Kv_temp + omega(p) * KH(i,:,p);
    end%对于每个核 p，通过 omega(p) 进行加权，将核矩阵中的每个矩阵 KH(i, :, p) 与核权重 omega(p) 相乘，然后将结果累加到 Kv_temp。
    
    ft = -(2*alpha*Kstar(i,:) + Kv_temp);%计算节点 i 的标签更新值 ft，它结合了图谱矩阵 Kstar 和加权核矩阵 Kv_temp。alpha 控制了图谱矩阵的影响权重。
    %2 * alpha * Kstar(i,:)：表示节点 i 与其他节点的图谱矩阵的影响。
    %Kv_temp 是加权后的核矩阵，表示节点 i 与其他节点的加权关系。
    Z_hat = -ft/2/(alpha+gamma(i));%Z_hat 是根据 ft 和正则化参数 alpha、gamma(i) 计算得到的拟合标签。gamma(i) 是节点 i 的正则化参数，调整了标签更新的幅度。
    
    indx = 1:num;%indx 是除了节点 i 之外的所有节点的索引。
    indx(i) = [];
    [Z(i,indx), ~] = EProjSimplex_new(Z_hat(:,indx));%EProjSimplex_new 是一个外部函数，用于将 Z_hat 投影到单纯形上（即确保标签的和为 1 且每个标签大于等于 0）。这一步是确保每个节点的簇标签满足非负且和为 1 的条件。
end

end

