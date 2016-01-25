function C = poolmax4(S) % max-pool and downsample by a factor 2
m1=bsxfun(@max,S(1:2:size(S,1)-1 , 1:2:size(S,2)-1)    ,   S(2:2:size(S,1) , 1:2:size(S,2)-1));
m2=bsxfun(@max,S(1:2:size(S,1)-1 , 2:2:size(S,2)  )    ,   S(2:2:size(S,1) , 2:2:size(S,2)  ));
C=bsxfun( @max,m1,m2);
end

