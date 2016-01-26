%% Backprop on 2 convolutional layers with 2 pooling layers and one fully connected layer at the end.

%% load training data from files.
images = (loadMNISTImages('train-images.idx3-ubyte')-0.5)*2;
labels = loadMNISTLabels('train-labels.idx1-ubyte');
onehot = MNISTLabels_to_onehot(labels,60000)*2-1;

%% Architecture
nrvis=28*28;
nrout=10;

W1=25^(-1/2)*randn(5,5,4);
featuremapsL2=zeros(24,24,4);
pooledmapsL2=zeros(12,12,4);

W2=9^(-1/2)*randn(3,3,4);
featuremapsL3=zeros(10,10,4);
pooledmapsL3=zeros(5,5,4);

W3=100^(-1/2)*randn(nrout,100);

%% go through training set
display('Training net... ');
tic;
for n=1:60000
    t=onehot(:,n);
    y1=reshape(images(:,n),[28,28]);
    
    %% forward prop
    for i=1:4
        featuremapsL2(:,:,i)=conv2(y1,W1(:,:,i),'valid'); %four maps 24x24 in size (image 28x28, kernel 5x5)
        pooledmapsL2(:,:,i)=poolavg4(featuremapsL2(:,:,i)); %four maps 12x12 in size (downsampling factor 2)
        y2=sum(pooledmapsL2,3); %sum together along dimension 3
        %optional activation function here
    end
    for i=1:4
        featuremapsL3(:,:,i)=conv2(y2,W2(:,:,i),'valid'); %four maps 10x10 in size (image 12x12, kernel 3x3)
        pooledmapsL3(:,:,i)=poolavg4(featuremapsL3(:,:,i)); %four maps 5x5 in size (downsampling factor 2)
        %y3=sum(pooledmapsL3,3);
        %optional activation function here
    end
    z4=W3*pooledmapsL3(:);
    y4=adjtanh(z4);
    
    %% derivative backprop
    dEdz4 = (y4-t).*der_adjtanh(z4); % error^2/2 loss function
    dEdy3 = W3'*dEdz4;
    dEdpmapsL3 = reshape(dEdy3,[5,5,4]);
    dEdfmapsL3 = upsample2(dEdpmapsL3*1/4); %average pooling over 4 units was used
    %derivative of optional activation function here
    dEdfmapsL3summed=sum(dEdfmapsL3,3);
    for i=1:4
        dEdpmapsL2(:,:,i) = filter2(dEdfmapsL3summed,W2(:,:,i),'full'); %propagate error
        dEdW2(:,:,i) = conv2(y2,dEdfmapsL3(:,:,i),'valid'); %calculate gradient also
    end
    dEdfmapsL2=upsample2(dEdpmapsL2);
    for i=1:4
        dEdW1(:,:,i)=conv2(y1,dEdfmapsL2(:,:,i),'valid'); %no need to propagate error here
    end
    
    
    if (mod(n,100)); continue; end;
    display(['training image: ', num2str(n), ' of 60000.']);
end
display(['time taken: ',num2str(floor(toc)), ' seconds.']);