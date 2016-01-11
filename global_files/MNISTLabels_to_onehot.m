function onehot = MNISTLabels_to_onehot(labels,nr)
onehot=zeros(10,nr);
for n=1:nr
    onehot(labels(n)+1,n)=1;
end
end
