function y = st_sigmoid(z) %Stochastic sigmoid activation function
y=rand(length(z),1) < 1./(1+exp(-z)); 
end

