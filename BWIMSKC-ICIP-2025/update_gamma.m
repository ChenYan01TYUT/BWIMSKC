
function [W, gamma] = update_gamma(KH, k,omega,Kstar,alpha) %该函数接收五个输入参数：KH：包含多个核的三维矩阵（每个核是一个矩阵）。k：每个节点的邻居数（用来控制局部结构的大小）。omega：每个核的权重。Kstar：当前的图谱矩阵。alpha：正则化参数。
%函数返回两个输出：W：每个节点的邻接矩阵，表示节点之间的相似度。gamma：一个向量，包含了每个节点的 gamma 值。
[num, num, numker] = size(KH); %获取输入的核矩阵 KH 的维度，其中 num 是数据点的数量（每个核的维度），numker 是核的数量（即核的个数）。

W = zeros(num);%初始化 W 为零矩阵，W(i, :) 表示节点 i 的邻接信息。
gamma_temp = zeros(num,1);%初始化 gamma_temp 为零向量，用于存储每个节点的 gamma 值。
D = zeros(1,num);%初始化 D 为零向量，用于存储每个节点与其他节点之间的距离或相似度度量。

for i = 1:num%计算每个节点的距离信息 Kv_temp：
    Kv_temp = zeros(1,num);%初始化 Kv_temp 为零向量，它用于存储节点 i 与其他所有节点的距离信息。
    for p = 1:numker
        Kv_temp = Kv_temp + omega(p) * KH(i,:,p);%对每个核 p，根据核的权重 omega(p) 和核矩阵 KH(i, :, p) 计算节点 i 与所有其他节点的距离。最终的 Kv_temp 是所有核加权的距离总和。
    end
    D = -(2*alpha*Kstar(i,:) + Kv_temp);%D 是节点 i 与所有其他节点的“局部结构”度量，包含图谱矩阵 Kstar 和加权核矩阵的贡献。通过 alpha 控制图谱矩阵的影响权重。这个公式反映了图谱学习中的平衡，试图通过调整 Kstar 和 Kv_temp 来捕捉局部结构。
    [dumb, idx] = sort(D, 2); %对 D 中的元素按值进行排序，dumb 是排序后的值，idx 是对应的索引。idx 将记录每个节点与其他节点的相似度排序。

    id = idx(1,2:k+2); %获取与节点 i 最相似的 k+2 个节点的索引（跳过自己，因为 idx(1, 1) 是节点 i 本身）。
    di = D(1, id); %id 是这些节点的索引，di 是这些节点的相似度或距离值。
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps); %计算邻接矩阵 W(i, id) 的值，它表示节点 i 与它的 k 个邻居之间的相似度，采用了一种标准化的方法。eps 是一个小常数，用于避免除零错误。
    gamma_temp(i) = k/2*di(k+1) - 1/2*sum(di(1:k)) - alpha;%计算 gamma_temp(i)，即节点 i 的 gamma 值。gamma 是图谱学习中的正则化参数，反映了节点的局部结构。
end;
gamma = gamma_temp;%循环结束后，更新 gamma 为 gamma_temp，并返回最终的邻接矩阵 W 和 gamma。







