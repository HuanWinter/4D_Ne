% DEPRECATED (2026-05-24): ported to Python in Src/mi_kraskov.py
%   Python `mi_kraskov(X, Y, method="exact")` matches this function to ~1e-11
%   (verified against this file via MATLAB R2024b). Kept for reference only.
function [ I1] = MI_Kraskov( X, Y, varargin )
%KraskovMI computes the Kraskov estimator for the mutual information.
%   1. Input: X, Y
%             zeroFix (optional): fix the negative estimation to 0 (default
%                                 false);
%
%   univariate: X, Y (n x 1) vector
%   multivariate: X, Y (n x m) matrix (rows=observations,
%   columns=variables)
%
%   2. Output: I1, I2: the two estimator of MI, I(1), I(2) (see Ref.)
%
% Ref: Kraskov, Alexander, Harald Stï¿½gbauer, and Peter Grassberger.
%      "Estimating mutual information." Physical review E 69.6 (2004): 066138.
%
% Author: Paolo Inglese <paolo.ingls@gmail.com>
% Last revision: 17-05-2015

k=1;

if nargin < 2 || nargin > 43
    error('Wrong input number.');
end
if nargin == 2
    zeroFix = false;
end
if nargin == 3
    if ~islogical(varargin{1})
        error('zeroFix must be true or false');
    else
        zeroFix = varargin{1};
    end
end
    

if size(X, 1) ~= size(Y, 1)
    error('X and Y must contain the same number of samples');
end

nObs = size(X, 1);
nx = zeros(nObs, 1);
ny = zeros(nObs, 1);

[X,Xindex]=sort(X,'ascend');
Y=Y(Xindex);
[Ysort,Yindex]=sort(Y,'ascend');
%inverseperm
Yinvind=[];
Yinvind(Yindex)=[1:nObs]; %Yinvind is such that Ysort(Yinvind)==Y

i=1;
Eps=Inf;
right=i+1;
xincr=0;
while(xincr<Eps &  right<=nObs)
    dxRight=abs(X(right)-X(i));
    dyRight=abs(Y(right)-Y(i));
    Eps=min(Eps,max(dxRight,dyRight));
    xincr=dxRight;
    right=right+1;
end



%  nx(i)=(sum(abs(X-X(i))<Eps)-1);
%  ny(i)=(sum(abs(Y-Y(i))<Eps)-1);
%%%%%%%%%%%%%%%%%%%%%%
sx=0;
for j=i+1:nObs
   if (abs(X(j)-X(i))<Eps)
   sx=sx+1;
   else
       break
   end
end
for j=i-1:-1:1
    if (abs(X(j)-X(i))<Eps)
    sx=sx+1;
    else
        break
    end
end
sy=0;
for j=Yinvind(i)+1:nObs
    if (abs(Ysort(j)-Y(i))<Eps)
    sy=sy+1;
    else
        break
    end
end
for j=Yinvind(i)-1:-1:1
   if (abs(Ysort(j)-Y(i))<Eps)
       sy=sy+1;
   else
       break
   end
end
% if (nx(i)~=sx)
%     keyboard
% elseif (ny(i)~=sy)
%     keyboard
% end
nx(i)=sx;
ny(i)=sy;
%%%%%%%%%%%%%%%%%%%%%%%%




for i=2:nObs-1
    Eps=Inf;
    left=i-1;
    xincr=0;
    while(xincr<Eps &  left>=1)
     dxLeft=abs(X(left)-X(i));
     dyLeft=abs(Y(left)-Y(i));
     Eps=min(Eps,max(dxLeft,dyLeft));
     xincr=dxLeft;
     left=left-1;
    end
    right=i+1;
    xincr=0;
    while(xincr<Eps &  right<=nObs)
     dxRight=abs(X(right)-X(i));
     dyRight=abs(Y(right)-Y(i));
     Eps=min(Eps,max(dxRight,dyRight));
     xincr=dxRight;
     right=right+1;
     end
% nx(i)=(sum(abs(X-X(i))<Eps)-1);
% ny(i)=(sum(abs(Y-Y(i))<Eps)-1);
%%%%%%%%%%%%%%%%%%%%%%%%
sx=0;
for j=i+1:nObs
   if (abs(X(j)-X(i))<Eps)
   sx=sx+1;
   else
       break
   end
end
for j=i-1:-1:1
    if (abs(X(j)-X(i))<Eps)
    sx=sx+1;
    else
        break
    end
end
sy=0;
for j=Yinvind(i)+1:nObs
    if (abs(Ysort(j)-Y(i))<Eps)
    sy=sy+1;
    else
        break
    end
end
for j=Yinvind(i)-1:-1:1
   if (abs(Ysort(j)-Y(i))<Eps)
       sy=sy+1;
   else
       break
   end
end
nx(i)=sx;
ny(i)=sy;
%%%%%%%%%%%%%%%%%%%%%%%%%



end

  i=nObs;     
  Eps=Inf;
  left=i-1;
  xincr=0;
    while(xincr<Eps &  left>=1)
     dxLeft=abs(X(left)-X(i));
     dyLeft=abs(Y(left)-Y(i));
     Eps=min(Eps,max(dxLeft,dyLeft));
     xincr=dxLeft;
     left=left-1;
   end    
% nx(i)=(sum(abs(X-X(i))<Eps)-1);
% ny(i)=(sum(abs(Y-Y(i))<Eps)-1);
%%%%%%%%%%%%%%%%%%%%%%%%
sx=0;
for j=i+1:nObs
   if (abs(X(j)-X(i))<Eps)
   sx=sx+1;
   else
       break
   end
end
for j=i-1:-1:1
    if (abs(X(j)-X(i))<Eps)
    sx=sx+1;
    else
        break
    end
end
sy=0;
for j=Yinvind(i)+1:nObs
    if (abs(Ysort(j)-Y(i))<Eps)
    sy=sy+1;
    else
        break
    end
end
for j=Yinvind(i)-1:-1:1
   if (abs(Ysort(j)-Y(i))<Eps)
       sy=sy+1;
   else
       break
   end
end
nx(i)=sx;
ny(i)=sy;
%%%%%%%%%%%%%%%%%%%%%%%%%


% mutual information estimators
I1 = psi(k) - sum(psi(nx + 1) + psi(ny + 1)) / nObs + psi(nObs);

if (zeroFix)
    if I1 < 0
        warning('First estimator is negative -> 0');
        I1 = 0;
    end

end

end