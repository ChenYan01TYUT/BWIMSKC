function [x,ft] = EProjSimplex_new(v, k)%定义了一个输入输出为 x 和 ft 的函数，v 是输入向量，k 是参数。x 是最终的优化结果，ft 是函数迭代的次数。
%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%

if nargin < 2%如果输入参数少于两个（即没有提供 k），则将 k 默认设置为 1。
    k = 1;
end;

ft=1;%ft = 1：初始化迭代次数为 1。
n = length(v);%n = length(v)：获取输入向量 v 的维度。

v0 = v-mean(v) + k/n;%v0 是对输入向量 v 进行调整后得到的向量，首先将 v 的均值减去，然后加上一个调整项 k/n，这使得投影可以适应不同的 k 值。
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;%计算 v0 中的最小值 vmin，如果最小值小于 0，则进入一个迭代过程。
    lambda_m = 0;%lambda_m 是一个用于调整的参数，初始值为 0。
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end;
        %进入一个迭代过程，当误差 f 的绝对值大于 10^-10 时继续迭代。f 是当前迭代的目标函数的误差。
        %在每次迭代中，v1 = v0 - lambda_m 是对 v0 进行修正，lambda_m 调整了 v1。
        %posidx 是一个逻辑向量，表示哪些元素大于零。
        %npos 计算出 v1 中大于零的元素的数量。
        %g = -npos 是梯度的负方向。
        %f = sum(v1(posidx)) - k 计算出当前修正后的 v1 向量的目标函数值（即与 k 的差距）。
        %lambda_m 更新，根据梯度 g 和误差 f 来调整。
        %通过 ft = ft + 1 记录迭代次数，最多进行 100 次迭代，如果超过，则退出循环并返回结果。
        %如果循环终止，x = max(v1, 0) 确保返回的结果是非负的。
    end;
    x = max(v1,0);%最后，将 v1 中小于零的元素置为 0，确保返回的向量 x 是非负的。

else
    x = v0;%如果 vmin >= 0，则直接返回 v0，不需要进一步的调整。
end;