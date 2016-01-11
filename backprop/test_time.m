%% Variables
images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
onehot=MNISTLabels_to_onehot(labels,10000);

correct=zeros(10000,1);
%%
tic
for n=1:10000
    t=onehot(:,n)*2-1;
    y1=(images(:,n)-0.5)*2;
    z2=W1*y1+b2;
    y2=adjtanh(z2);
    z3=W2*y2+b3;
    y3=adjtanh(z3);
    
    [val,ind]=max(y3); %classified index
    [valt,indt]=max(t); %target index
    if(ind==indt)
        correct(n)=1;
    end
end
display(['time taken: ',num2str(floor(toc)), ' seconds'])
display(['#errors: ',num2str(10000-sum(correct(:)))]);
display(['error rate: ',num2str(1-mean(correct(:)))]);
