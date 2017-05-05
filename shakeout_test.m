w1 = load('test_data/cvalue05.mat');
w2 = load('test_data/cvalue10.mat');
w3 = load('test_data/cvalue50.mat');

[mean_c05,std_c05] = stats_norms(w1);
[mean_c10,std_c10] = stats_norms(w2);
[mean_c50,std_c50] = stats_norms(w3);


function [mean_w0, std_w0]=stats_norms(norm_data)
  norm_w0 = reshape(norm_data.weights.fc1,[],1);
  mean_w0 = mean(norm_w0);
  std_w0 = std(norm_w0);
end
