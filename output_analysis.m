function [precision, recall, f_measure,accuracy] =  output_analysis(bin_GT, bin_alg)

add_m=bin_alg+bin_GT;
tp=sum(add_m(:)==2);
tn=sum(add_m(:)==0);

sub_m=bin_alg-bin_GT;
fp=sum(sub_m(:)==1);
fn=sum(sub_m(:)==-1);

precision=tp/(tp+fp);
recall=tp/(tp+fn);
f_measure=2*precision*recall/(precision+recall);
accuracy=(tp+tn)/(tp+tn+fp+fn);