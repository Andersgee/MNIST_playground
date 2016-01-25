function y = adjtanh(z) %adjusted tanh function so that adjtanh(1)=1. Recommended in (Tricks of the Trade, Lecun et al., 1998)
y=1.7159*tanh(2/3*z);
end