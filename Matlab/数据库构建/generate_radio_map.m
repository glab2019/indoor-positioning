function fingerprint = generate_radio_map(grid_size)
% ���ɡ�RSS���滷�����ݼ���
% ���ص�fingerprint��һ����ά���飬��¼�������RSSֵ����һά�͵ڶ�ά�Ƿ���ĳߴ磬����ά����ͬ��AP��
    %% ��������
    if nargin == 0
        grid_size = 0.01;
    end
    room_x = 20;
    room_y = 15;
    room_z = 4;
    f = 2400; %�ź�Ƶ��
    % ����AP��λ��
    APs = [1,1 
        1,14
        4, 4
        4, 8
        4, 12
        8, 4
        8, 8
        8, 12
        12, 4
        12, 8
        12, 12
        16, 4
        16, 8
        16, 12
        19,1
        19,14
    ];
    %% ����fingerprint
    fingerprint = zeros(room_x / grid_size -1, room_y / grid_size -1, size(APs, 1));
    for i = 1 : size(APs, 1)
        source_x = APs(i, 1);
        source_y = APs(i, 2);
        source_z = 1.5; %Ĭ���ź�Դ�ĸ߶�Ϊ2m
        rss = get_rss_by_ray_tracing(room_x, room_y, room_z, source_x, source_y, source_z, grid_size, f); %�������߸��ټ���RSS
        fingerprint(:, :, i) = rss;
%         figure;
%         mesh(fingerprint(:, :, i));
    end
    save('radio_map_20_15', 'fingerprint');
end

