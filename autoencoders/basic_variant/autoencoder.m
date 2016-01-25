% Unsupervised 3-layer net trained to output its input via a hidden layer.

%% load training data from files.
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

%% Architechture
nrvis=28*28;
nrhid=7^2;
nrout=28*28;

y1=zeros(nrvis,1); %Layer 1
W1=nrvis^(-1/2)*randn(nrhid, nrvis);
z2=zeros(nrhid,1); y2=zeros(nrhid,1); %Layer 2
W2=nrhid^(-1/2)*randn(nrout, nrhid);
z3=zeros(nrout,1); y3=zeros(nrout,1); %Layer 3

%% setup plots
subplot(2,2,1);
inputplot=imshow(zeros(28,28),[-1 1]); axis off;
subplot(2,2,2);
W1plot=imshow(zeros(sqrt(nrvis)*sqrt(nrhid),sqrt(nrvis)*sqrt(nrhid)),[-0.5 0.5]); axis off; title('encoder (W1)');
subplot(2,2,3);
W2plot=imshow(zeros(sqrt(nrout)*sqrt(nrhid),sqrt(nrout)*sqrt(nrhid)),[-0.5 0.5]); axis off; title('decoder (W2)');
subplot(2,2,4)
outputplot=imshow(zeros(28,28),[-1 1]); axis off;

%% go through training set
tic
for n=1:60000
    y1=(images(:,n)-0.5)*2;
    t=y1;
    z2=W1*y1+b2;
    y2=adjtanh(z2);
    z3=W2*y2+b3;
    y3=adjtanh(z3);

    dEdW2=[(y3-t).*der_adjtanh(z3)] * [y2]';
    dEdW1=[ (W2'*((y3-t).*der_adjtanh(z3))) .* der_adjtanh(z2) ] * [y1]';
    W2=W2-0.001*dEdW2;
    W1=W1-0.001*dEdW1;

    if (mod(n,300)); continue; end;
    set(inputplot, 'CData',reshape(y1,[28,28]));
    set(W1plot, 'CData',rearrangeW(W1));
    set(W2plot, 'CData',rearrangeW(W2'));
    set(outputplot, 'CData',reshape(y3,[28,28]));
    drawnow;
end
display(['time taken: ',num2str(floor(toc)), ' seconds'])
