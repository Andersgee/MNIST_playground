%Restricted Boltzman Machine with Contrastive Divergence learning (1 full step)
nrvis=28*28;
nrhid=10^2;
v_0=zeros(nrvis,1);
h_0=zeros(nrhid,1);
v_1=zeros(nrvis,1);
h_1=zeros(nrhid,1);
W=0.01*randn(nrhid, nrvis);
Wplot=imshow(zeros(sqrt(nrvis)*sqrt(nrhid),sqrt(nrvis)*sqrt(nrhid)),[-2 2]); axis off;
images = loadMNISTImages('train-images.idx3-ubyte');

tic
for n=1:60000
    v_0=images(:,n);
    h_0=st_sigmoid(W*v_0);
    v_1=st_sigmoid(W'*h_0);
    h_1=sigmoid(W*v_1);
    W=W+0.1*(h_0*v_0'-h_1*v_1');

    if (mod(n,300)); continue; end; %skip most plot updates
    set(Wplot, 'CData',rearrangeW(W)); drawnow;
end
display(['time taken: ',num2str(floor(toc)), ' seconds'])

%According to Hinton, which originally proposed this learning scheme as a fast
%approximation to maximum likelihood learning later explained that..
%in fact it is really just an autoencoder, and the learning rule is
%actually backprop (approximately) through stochastic units.
