load net_br.mat;
load('online_data.mat')
rss=rss(150:250,:);
trace=trace(150:250,:);
a=rss';
result=sim(net,a);
result=result';
figure; 
l=length(rss);
for i=1:l
h1=plot(trace(i, 1), trace(i, 2), 'g.');
hold on;
h2=plot(result(i, 1), result(i, 2), 'r*');
legend([h1(1),h2(1)],'实际点的位置', '预测点的位置');
title('预测的点的位置:*');
hold on;
pause(0.05)
end