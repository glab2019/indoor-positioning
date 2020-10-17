function [data, labels] = get_offline_data_random(fingerprint, data_num)
%ģ���������ݲɼ����������
    if nargin == 1
        data_num = 30000;  %Ĭ��30000������
    end
    [size_x, size_y, size_ap] = size(fingerprint);
    data = reshape(fingerprint, [], size_ap);
    [x, y] = meshgrid(1:size_x, 1:size_y);
    x = x';
    y = y';
    labels = [x(:), y(:)];
    
    %���������������ѡ��һ����
    idx = randperm(size(data, 1), data_num);
    data = data(idx, :);
    labels = labels(idx, :);
end