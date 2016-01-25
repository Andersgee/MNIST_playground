function map = complexlogmap(image, focus, ppu)
%map is the interpolated complex logarithm (log-polar) mapping of image
%focus is the array [y;x] determining where focus is
%ppu is pixels per unit on the interpolated map
if ~exist('ppu','var') ppu=10; end

r = sqrt(size(image,1)^2+size(image,2)^2); %diameter of image
[u,v] = meshgrid(0: 1/ppu :log(r)  ,  -pi+1/ppu: 1/ppu :pi);
xqyqi=exp(u+v*1i); %query points to interpolate relative to focus
map = interp2(image, focus(2)+real(xqyqi) , focus(1)+imag(xqyqi) ,'cubic');
map(isnan(map))=0; %points outside image data = 0 instead of NaN

end

