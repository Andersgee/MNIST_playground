%% Variables
nrvis=28*28;
nrhid=49;
v_0=zeros(nrvis,1);
h_0=zeros(nrhid,1);
v_1=zeros(nrvis,1);
h_1=zeros(nrhid,1);
W=0.01*randn(nrhid, nrvis);
Wplot=imshow(zeros(sqrt(nrvis)*sqrt(nrhid),sqrt(nrvis)*sqrt(nrhid)),[-2 2]); axis off;
images = loadMNISTImages('train-images.idx3-ubyte');

tic
for n=1:60000
    v_0=images(:,n);                     %visible units (data vector)
    h_0=st_sigmoid(W*v_0);               %hidden units (stochastic sigmoid)
    v_1=st_sigmoid(W'*h_0);              %reconstructed visible units (stochastic sigmoid)
    h_1=sigmoid(W*v_1);                  %reconstructed hidden units (sigmoid)
    W=W+0.1*(h_0*v_0'-h_1*v_1');         %Change Weights. Increase when looking at data, decrease when looking at reconstruction.

    if (mod(n,200)); continue; end;      %Skip updating plot for most iterations.
    set(Wplot, 'CData',rearrangeW(W)); drawnow;
end
display(['time taken: ',num2str(floor(toc)), ' seconds'])