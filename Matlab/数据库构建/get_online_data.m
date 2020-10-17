function [ trace, rss ] = get_online_data( fingerprint, gridSize, roomLength, roomWidth, t )
%�õ�����λ��ָ�Ʒ�������
%���룺rss���滷�����ݼ���������ݼ���gridSize������ߴ磬�������ݵĸ���
%�����λ�õ�켣���Լ��켣��ÿ�����ϵ�rss
    trace = get_random_trace(roomLength, roomWidth, t);
    rss = zeros(size(trace, 1), size(fingerprint, 3));
    for i = 1 : size(trace, 1);
        x = round(trace(i, 1) / gridSize);
        y = round(trace(i, 2) / gridSize);
        if x < 1
            x = 1;
        elseif x > size(fingerprint, 1)
            x = size(fingerprint, 1);
        end
        if y < 1
            x = 1;
        elseif y > size(fingerprint, 2)
            y = size(fingerprint, 2);
        end
        rss(i, :) = fingerprint(x, y, :);
        trace(i, :) = [x, y];
    end
end

