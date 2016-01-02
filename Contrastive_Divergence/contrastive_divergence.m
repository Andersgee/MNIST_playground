%% Variables
v_0=zeros(28*28,1);
h_0=zeros(49,1);
v_1=zeros(28*28,1);
h_1=zeros(49,1);
W=0.01*randn(49, 28*28);
Wplot=imshow(zeros(28*7,28*7),[-2 2]); axis off;
images = loadMNISTImages('train-images.idx3-ubyte');

tic
%% Iterate over all images
for n=1:60000
    v_0=images(:,n);                %visible units (data vector)
    h_0=stsig(W*v_0);               %hidden units (stochastic sigmoid activation function)
    v_1=stsig(W'*h_0);              %reconstructed visible units (stochastic sigmoid)
    h_1=sig(W*v_1);                 %reconstructed hidden units (normal sigmoid)
    W=W+0.1*(h_0*v_0'-h_1*v_1');    %Change Weights using covariance matrix. Increase weights when looking at data, decrease when reconstructing.

    if (mod(n,200)); continue; end;  %Skip updating plot for most iterations. faster.
    %reshape Weightmatrix (from 49x784 to 196x196) so features can be visualized. Not necessary but fun to see :)
    featureRows=mat2cell(reshape(W',[28,28*49]),[28],[ones(1,7)*28*7]);
    set(Wplot, 'CData',cat(1,featureRows{:})); drawnow;
end
display(['time taken: ',num2str(floor(toc)), ' seconds'])