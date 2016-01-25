%% Variables
nrvis=28*28;
nrhid=10^2;
nrout=28*28;
y1=zeros(nrvis,1);
z2=zeros(nrhid,1);
b2=zeros(nrhid,1);
y2=zeros(nrhid,1);
z3=zeros(nrout,1);
b3=zeros(nrout,1);
y3=zeros(nrout,1);
W1=0.01*randn(nrhid, nrvis);
W2=W1';
%W2=0.01*randn(nrout, nrhid);

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
%onehot=MNISTLabels_to_onehot(labels,60000);

subplot(2,2,1);
inputplot=imshow(zeros(28,28),[-1 1]); axis off;
subplot(2,2,2);
W1plot=imshow(zeros(sqrt(nrvis)*sqrt(nrhid),sqrt(nrvis)*sqrt(nrhid)),[-0.5 0.5]); axis off; title('encoder W, decoder W^T');
subplot(2,2,3);
y2plot=imshow(zeros(sqrt(nrhid),sqrt(nrhid)),[-1.7 1.7]); axis off; title('code (activities of hidden)');
subplot(2,2,4)
outputplot=imshow(zeros(28,28),[-1 1]); axis off;

%%
tic
for lr=1:5
for n=1:60000
    y1=(images(:,n)-0.5)*2;
    t=y1;
    
    y1=y1 .* (rand(28*28,1)>0.5); %gaussian noise and dropout
    z2=W1*y1+b2;
    y2=adjtanh(z2);
    z3=W1'*y2+b3;
    y3=adjtanh(z3);
    
    dEdW2=[(y3-t).*der_adjtanh(z3)] * [y2]';
    dEdW1=[ (W1*((y3-t).*der_adjtanh(z3))) .* der_adjtanh(z2) ] * [y1]';
    W1=W1-0.001/lr*dEdW1;
    W1=W1-[0.001/lr*dEdW2]';
    
    %squared_error=sum((t(:)-y3(:)).^2)/2
    
    if (mod(n,300)); continue; end;
    set(inputplot, 'CData',reshape(y1,[28,28]));
    set(W1plot, 'CData',rearrangeW(W1));
    set(y2plot, 'CData',reshape(y2,[sqrt(nrhid),sqrt(nrhid)]));
    set(outputplot, 'CData',reshape(y3,[28,28]));
    drawnow;
end
lr
end
display(['time taken: ',num2str(floor(toc)), ' seconds'])