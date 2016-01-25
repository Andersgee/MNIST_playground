%% load test data
images_t = loadMNISTImages('t10k-images.idx3-ubyte');
labels_t = loadMNISTLabels('t10k-labels.idx1-ubyte');
images_t=(images_t-0.5)*2;
images_t=images_t/std(images_t(:));
onehot_t=MNISTLabels_to_onehot(labels_t,10000)*2-1;

display('Evaluating net... ');
correct=zeros(10000,1);
for n=1:10000
    t=onehot_t(:,n);
    y1=images_t(:,n);
    
    z2=W1*y1;
    y2=adjtanh(z2);
    z3=W2*y2;
    y3=adjtanh(z3);

    [val,ind]=max(y3); %classified index
    [valt,ind_t]=max(t); %target index
    if(ind==ind_t)
        correct(n)=1;
    end
end
display(['error rate: ',num2str(1-mean(correct(:)))]);
