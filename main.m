
function [ W ]  = SCDRMFS( X_train, Y_train, para)
%SCDRMFS 此处显示有关此函数的摘要
%   此处显示详细说明
[num_train, num_feature] = size(X_train); 
num_label = size(Y_train, 2);
%% Initialize 
iter=1;

c = round(para.c);
V= rand(num_train, c); 
W = rand(num_feature, c); 
Q= rand(c,num_feature); 
P = rand(c, num_label); 

I = eye(num_train) ;
H = I - 1 / num_train * ones(num_train, 1) * ones(num_train, 1)';
%Update --------------------------------------------------------------

%% calculate L,Lx,LV
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 1;

S = constructW(V, options);
A=diag(sum(S,2));
L = A-S;

SX = constructW(X_train', options); 
AX=diag(sum(SX,2));
LX = AX-SX;

SV = constructW(V', options);
AV=diag(sum(SV,2));
LV = AV-SV;


    
X=X_train;
Y=Y_train;
%% U
U = update_U(W);

    
while iter<=10
  
 %% W
    W1=X'*H*V+para.gamma*(SX*W+W*SV);
    W2 = X'*H*X*W+para.gamma*(AX*W+W*AV)+para.lambda*U*W;
    for i=1:num_feature
        for j =1:c
            W(i,j) = W(i,j)*(W1(i,j)/W2(i,j));
        end
    end
	U = update_U(W);
%% V
    V1=H*X*W+S*V+para.alpha*X*Q'+para.beta*Y*P';
    V2=H*V+A*V+para.alpha*V*Q*Q'+para.beta*V*P*P';
    for i=1:num_train
        for j =1:c
            V(i,j) = V(i,j)*(V1(i,j)/V2(i,j));
        end
    end
  
%% Q
     Q1 =V'*X;
     Q2 = V'*V*Q;
    for i=1:c
        for j =1:num_feature
            Q(i,j) = Q(i,j)*(Q1(i,j)/Q2(i,j));
        end
    end
   
%% P
     P1 =V'*Y;
     P2 = V'*V*P;
    for i=1:c
        for j =1:num_label
            P(i,j) = P(i,j)*(P1(i,j)/P2(i,j));
        end
    end
    iter = iter + 1;
end

end


