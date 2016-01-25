%% Variables
nrout=10;
y1=zeros(nrvis,1);
z2=zeros(nrhid,1);
b2=zeros(nrhid,1);
y2=zeros(nrhid,1);
z3=zeros(nrout,1);
b3=zeros(nrout,1);
y3=zeros(nrout,1);
%W1 is pretrained. learn a new W2 with 10 output units
W2=0.01*randn(nrout, nrhid);

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
onehot=MNISTLabels_to_onehot(labels,60000);

subplot(2,2,1);
inputplot=imshow(zeros(28,28),[-1 1]); axis off;
subplot(2,2,2);
W1plot=imshow(zeros(sqrt(nrvis)*sqrt(nrhid),sqrt(nrvis)*sqrt(nrhid)),[-1 1]); axis off; title('W1');
subplot(2,2,3);
W2plot=imshow(zeros(sqrt(nrhid),sqrt(nrhid)*nrout),[-1 1]); axis off; title('W2');
subplot(2,2,4)
outputplot=imshow(zeros(10,2),[-1.7 1.7]); axis off; title('output, target');

%%
tic
for lr=1:6
for n=1:60000
    t=onehot(:,n)*2-1;
    y1=(images(:,n)-0.5)*2;
    z2=W1*y1+b2;
    y2=adjtanh(z2);
    z3=W2*y2+b3;
    y3=adjtanh(z3);
    
    dEdW2=[(y3-t).*der_adjtanh(z3)] * [y2]';
    dEdW1=[ (W2'*((y3-t).*der_adjtanh(z3))) .* der_adjtanh(z2) ] * [y1]';
    W2=W2 - 0.001/lr * dEdW2;
    W1=W1 - 0.001/lr * dEdW1;
    
    %squared_error=sum((t(:)-y3(:)).^2)/2
    
    if (mod(n,600)); continue; end;
    set(inputplot, 'CData',reshape(y1,[28,28]));
    set(W1plot, 'CData',rearrangeW(W1));
    W2reshaped=reshape(W2',[sqrt(nrhid)],[sqrt(nrhid)*10]);
    set(W2plot, 'CData',W2reshaped);
    set(outputplot, 'CData',[y3,t]);
    drawnow;
end
lr
end
display(['time taken: ',num2str(floor(toc)), ' seconds'])