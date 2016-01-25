% Same as backprop_basic but includes visualizations
% of activity during training. //Anders

%% load and setup training data.
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images=(images-0.5)*2;
images=images/std(images(:));
onehot = MNISTLabels_to_onehot(labels,60000)*2-1;

%% Architecture
nrvis=28*28;
nrhid=5^2;
nrout=10;

y1=zeros(nrvis,1); %Layer 1
W1=nrvis^(-1/2)*randn(nrhid, nrvis);
z2=zeros(nrhid,1); y2=zeros(nrhid,1); %Layer 2
W2=nrhid^(-1/2)*randn(nrout, nrhid);
z3=zeros(nrout,1); y3=zeros(nrout,1); %Layer 3

%% setup plots
subplot(1,5,1); inputplot=imshow(zeros(28,28),[-1 1]);                          axis off; title('input data');
subplot(1,5,2); W1plot=imshow(zeros([sqrt(nrvis)*nrhid],[sqrt(nrvis)]),[-1 1]); axis off; title('W1');
subplot(1,5,3); hiddenplot=imshow(zeros(nrhid,1),[-1.7 1.7]);                   axis off; title('hidden');
subplot(1,5,4); W2plot=imshow(zeros(sqrt(nrhid)*nrout,sqrt(nrhid)),[-1 1]);     axis off; title('W2');
subplot(1,5,5); outputplot=imshow(zeros(nrout,2),[-1.7 1.7]);                   axis off; title('output, target');

%% go through training set
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

    % update plots (skip most plot updates)
    if (mod(n,100)); continue; end;
    set(inputplot, 'CData',reshape(y1,[28,28]));
    set(W1plot, 'CData',(reshape(W1',[sqrt(nrvis)],[sqrt(nrvis)*nrhid]))');
    set(hiddenplot, 'CData',y2);
    set(W2plot, 'CData',(reshape(W2',[sqrt(nrhid)],[sqrt(nrhid)*nrout]))');
    set(outputplot, 'CData',[y3,t]);
    drawnow;
end
display(['time taken: ',num2str(floor(toc)), ' seconds.']);

evaluate;