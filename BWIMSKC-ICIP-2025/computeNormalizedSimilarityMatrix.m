function S = computeNormalizedSimilarityMatrix(K, numNeighbors)
    % 输入：
    % K - 相似度矩阵，大小为 nSmp x nSmp
    % numNeighbors - 每个点的邻居数，即 |N_i|
    
    [nSmp, ~] = size(K);  % 获取样本数
    
    % 初始化矩阵S
    S = zeros(nSmp, nSmp);
    
    % 遍历每个样本i
    for i = 1:nSmp
        % 找到第i个样本的前numNeighbors个最近邻（不包括自身）
        [~, idx] = sort(K(i, :), 'descend');  % 根据第i个样本与其他样本的相似度降序排列
        neighbors = idx(2:numNeighbors+1);  % 取前numNeighbors个邻居（排除自身，即idx(1)）
        
        % 计算归一化因子
        sumK = sum(K(i, neighbors));  % 计算与第i个样本的邻居的相似度之和
        
        % 遍历邻居，将相似度归一化并存入S
        for j = neighbors
            S(i, j) = K(i, j) / sumK;  % 按公式计算
        end
    end
end
