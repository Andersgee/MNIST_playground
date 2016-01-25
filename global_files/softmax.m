function p = softmax(y) %p is y values converted to probabilities with softmax

T=1; %high temp gives more even probabilities
total=sum(exp(y(1:length(y))/T));
p=exp(y(1:length(y))/T)/total;

end

