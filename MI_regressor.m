clear
clc
format long 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load data

vari_set = {'DST';
            'AE Index';
            'ap';
            'f10.7 index';
            'Kp';
            'Ve';
            'Bx';
            'By';
            'Bz'};
iteration = -8:7;

MI = zeros(length(iteration),length(vari_set));
MI_mean = zeros(length(iteration),length(vari_set));
MI_std = zeros(length(iteration),length(vari_set));

for i = 1:length(iteration)
    data = load(['data/Delay/all_',num2str(iteration(i)),'.mat']);
    
    Alt = data.out(:,1);
    mLat = data.out(:,2);

    varies = data.out(:,6:14);
    NmF2 = data.out(:,17);
    idx = find(~isnan(varies(:,1))...
               &  ~isnan(varies(:,2))...
               &  ~isnan(varies(:,3))...
               &  ~isnan(varies(:,4))...
               &  ~isnan(varies(:,5))...
               &  ~isnan(varies(:,6))...
               &  ~isnan(varies(:,7))...
               &  ~isnan(varies(:,8))...
               &  ~isnan(varies(:,9))...
               & mLat>60);
    
    
    parfor j = 1:length(varies(1,:))
        
        normA = varies(:,j) - min(varies(:,j));
        varies(:,j) = normA ./ max(normA(:));
    end
    
    for j = 1:length(vari_set)
        num_resample = 16;
        MI_t = zeros(num_resample,1);
        CMI_t = zeros(num_resample,1);
        
        ind_num = 1:floor(length(idx)*0.6);
        tic
        parfor k = 1:num_resample
            %tic
            idx_t = idx(randperm(length(idx)));
            X_t = varies(idx_t(ind_num),j);
            Y_t = log(NmF2(idx_t(ind_num)));
            MI_t(k) = MI_Kraskov(X_t,Y_t);
            %toc

        end
        disp(['i=', num2str(i), ', j=', num2str(j)])
        toc
        MI_mean(i,j) = mean(MI_t);
        MI_std(i,j) = std(MI_t);
        
        MI(i,j) = MI_Kraskov(varies(idx,j),log(NmF2(idx)));
    end
end