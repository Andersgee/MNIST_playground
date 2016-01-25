%same as backprop_basic but using gpuArray.

%% load and setup training data.
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images=(images-0.5)*2;
images=gpuArray(images/std(images(:)));
onehot = gpuArray(MNISTLabels_to_onehot(labels,60000)*2-1);

%% Architecture
nrvis=28*28;
nrhid=5^2;
nrout=10;

y1=gpuArray(zeros(nrvis,1)); %Layer 1
W1=gpuArray(nrvis^(-1/2)*randn(nrhid, nrvis));
z2=gpuArray(zeros(nrhid,1)); y2=gpuArray(zeros(nrhid,1)); %Layer 2
W2=gpuArray(nrhid^(-1/2)*randn(nrout, nrhid));
z3=gpuArray(zeros(nrout,1)); y3=gpuArray(zeros(nrout,1)); %Layer 3

%% go through the training set
display('Training net... ');
tic;
for n=1:60000
    t  = onehot(:,n);
    y1 = images(:,n);

    % forward prop
    z2 = W1*y1;
    y2 = adjtanh(z2);
    z3 = W2*y2;
    y3 = adjtanh(z3);

    % derivative backprop
    dEdz3 = (y3-t).*der_adjtanh(z3);
    dEdW2 = dEdz3 * y2';
    dEdz2 = (W2'*dEdz3).*der_adjtanh(z2);
    dEdW1 = dEdz2 * y1';

    % adjust weights
    W2=W2-0.001*dEdW2;
    W1=W1-0.001*dEdW1;
end
display(['time taken: ',num2str(floor(toc)), ' seconds.']);

evaluate;
