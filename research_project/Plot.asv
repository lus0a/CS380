clear; clc;
m=39;
n=m;
x = linspace(0,1,m);   
y = linspace(0,1,n);   
[X,Y] = meshgrid(x,y);

A = importdata('exact.txt');
x_exact = reshape(A,m,n);
mesh(X,Y,x_exact);
xlabel('x');ylabel('y');zlabel('U(x,y)');

figure;
B = importdata('err_CG_x_CPU.txt');
x_err_CG_x_CPU = reshape(B,m,n);
mesh(X,Y,x_err_CG_x_CPU);
xlabel('x');ylabel('y');zlabel('U(x,y)');

figure;
C = importdata('err_CG_x_GPU.txt');
x_err_CG_x_GPU = reshape(C,m,n);
mesh(X,Y,x_err_CG_x_GPU);
xlabel('x');ylabel('y');zlabel('U(x,y)');

figure;
D = importdata('err_GD_x_CPU.txt');
x_err_GD_x_CPU = reshape(D,m,n);
mesh(X,Y,x_err_GD_x_CPU);
xlabel('x');ylabel('y');zlabel('U(x,y)');

figure;
E = importdata('err_GD_x_GPU.txt');
x_err_GD_x_GPU = reshape(E,m,n);
mesh(X,Y,x_err_GD_x_GPU);
xlabel('x');ylabel('y');zlabel('U(x,y)');

