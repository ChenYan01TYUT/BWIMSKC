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

dataName = 'FACS';
disp(dataName);
load([path,'datasets/',dataName,'_v2'],'X','Y');
Y(Y==-1)=2;
numclass = length(unique(Y));
numNeighbors = 5;

[nSmp, mSmp] = size(X);
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
A = constructW(X,options);

DSsym = 1./sqrt(max(sum(A, 2), eps));
Gnorm = (DSsym * DSsym') .* A;
Gnorm = (Gnorm + Gnorm') / 2;
Gnorm = sparse(Gnorm);
L = speye(nSmp) - Gnorm;

KH_all = zeros(nSmp, nSmp, 4);

L_half = L / 2;
I_L_half = speye(nSmp, nSmp) - L_half;

L_p = speye(nSmp, nSmp); 

L_p_precomputed = zeros(nSmp, nSmp, 4);
L_p_precomputed(:,:,1) = L_p;
for pk_val = 1:3
    L_p_precomputed(:,:,pk_val+1) = L_p_precomputed(:,:,pk_val) * L_half; 
end

L_q_precomputed = zeros(nSmp, nSmp, 4); 
L_q_precomputed(:,:,1) = speye(nSmp, nSmp);  % (I - L/2)^0 = I
L_q_precomputed(:,:,2) = I_L_half;  % (I - L/2)^1 = I - L/2

for q_val = 2:3
    L_q_precomputed(:,:,q_val+1) = L_q_precomputed(:,:,q_val) * I_L_half;  % 
end

for i = 1:4
    L_q = L_q_precomputed(:,:,qk+1);  
    L_p = L_p_precomputed(:,:,pk+1);  
    B_value = beta(pk+1, qk+1);
    G = (L_p * L_q) / (2 * B_value);
    X_G = G*X;
    KH_all(:, :, i) = (G * X_G) * (X_G' * G');

    pk = pk + 1;
    qk = qk - 1;
end

%%
alpha_range = 2.^[0:1:10];

for alpha_indx = 1:length(alpha_range)
    alpha = alpha_range(alpha_indx);
    
    [Kstar,Z,gamma,omega,obj] = Graph_main(KH_all,alpha);
    
    Kstar = kcenter(Kstar);
    Kstar = knorm(Kstar);
    [H, ~] = eigs(Kstar, numclass, 'la');
    [res(:,alpha_indx)] = myNMIACCV2(H,Y,numclass);
    
    fprintf('ACC:%4.4f \t NMI:%4.4f \t Pur:%4.4f \t Rand:%4.4f \n',...
        [ res(1,alpha_indx),res(2,alpha_indx),res(3,alpha_indx),res(4,alpha_indx) ]);
end
%%
[~,max_indx] = max(res(1,:,:),[],'all','linear'); 
res_opt = res(:,max_indx)

fprintf('ACC_max:%4.4f \t NMI_max:%4.4f \t Pur_max:%4.4f \t Rand_max:%4.4f \t indx:%4.0f \n',...
    [ res(1,max_indx) res(2,max_indx) res(3,max_indx) res(4,max_indx) max_indx]);


figure;
plot(1:20, obj(1:20), 'o-', 'LineWidth', 2, 'Color', [0 0.447 0.741], 'MarkerFaceColor', [0 0.447 0.741], 'MarkerSize', 8);
xlabel('Iteration', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Objective Value', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 16, 'LineWidth', 1.5); 
xlim([1 20]); 
legend('Objective Value', 'Location', 'northeast'); 
hold off;
