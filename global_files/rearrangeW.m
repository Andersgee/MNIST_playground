function Wr = rearrangeW(W) %ASSUMES THAT sqrt(size(W,1))=integer
%rearranges Weightmatrix to show features.
s1=size(W,1); %nrhid
s2=size(W,2); %nrvis
frow=reshape(W',[sqrt(s2)],[sqrt(s2)*s1]);
frowc=mat2cell(frow,[sqrt(s2)],[ones(1,sqrt(s1))*sqrt(s2)*sqrt(s1)]);
Wr=cat(1,frowc{:});
end
