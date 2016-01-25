function C = poolavg4(S) % average-pool and downsample by a factor 2
m1=bsxfun(@plus,S(1:2:size(S,1)-1 , 1:2:size(S,2)-1)    ,   S(2:2:size(S,1) , 1:2:size(S,2)-1));
m2=bsxfun(@plus,S(1:2:size(S,1)-1 , 2:2:size(S,2)  )    ,   S(2:2:size(S,1) , 2:2:size(S,2)  ));
C=bsxfun( @plus,m1,m2)/4;
end

