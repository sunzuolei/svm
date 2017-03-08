function svm = svmTrain(X,Y,kertype,C)

n=50;
x1 = randn(2,n);
x2 = 5+randn(2,n);
y1 = ones(1,n);
y2 = -ones(1,n); 
X = [x1,x2];        
Y = [y1,y2];
C = 10;  

kertype = 'linear';

options = optimset;    % Options是用来控制算法的选项参数的向量
options.LargeScale = 'off'; %largescale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';  %这样设置意味着没有输出

n = length(Y);  %数组Y的长度
H = (Y'*Y).*kernel(X,X,kertype);
f = -ones(n,1); %f为1*n个-1,f相当于Quadprog函数中的c
A = [];
b = [];
Aeq = Y; %相当于Quadprog函数中的A1,b1
beq = 0;
lb = zeros(n,1); %相当于Quadprog函数中的LB，UB
ub = C*ones(n,1);
a0 = zeros(n,1);  % a0是解的初始近似值
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);

epsilon = 1e-8;                     
sv_label = find(abs(a)>epsilon);  %0<a<a(max)则认为x为支持向量     
svm.a = a(sv_label);
svm.Xsv = X(:,sv_label);
svm.Ysv = Y(sv_label);
svm.svnum = length(sv_label);



%svm.label = sv_label;
