% DEPRECATED (2026-05-24): ported to Python in Src/show_mi.py
%   Use: from show_mi import show_mi; show_mi(MI, MI_mean, MI_std, X, vari_set)
%   Kept for reference only.
function show_MI(MI,MI_mean,MI_std,X,vari_set)
%SHOW_MI Summary of this function goes here
%   Detailed explanation goes here
figure(100);
for i = 1:length(MI(1,:))
    subplot(3,3,i);
    plot(X,MI(:,i),'yo-','LineWidth',5);hold on;
    plot(X,MI_mean(:,i),'g*-');hold on;
    plot(X,MI_mean(:,i)-3*MI_std(:,i),'r--');hold on;
    plot(X,MI_mean(:,i)+3*MI_std(:,i),'b--');hold on;
    xlabel('Time gap(h)');
    legend('all','mean','upper','lower');
    title(['MI between NmF2 and ', vari_set(i)]);
end

end

