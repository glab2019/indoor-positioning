function Power_all = get_rss_by_ray_tracing(room_x, room_y, room_z, source_x, source_y, source_z, grid_size, f)
% ���߸��٣� ���뷿���С���ź�Դ�����꣬���������С���ź�Ƶ�ʣ����ÿ��������Ͻ��յ����ź�ǿ��
% ��������ֱ�Ϊ������ߴ�x�� y�� z���ź�Դ�����ߣ����꣨x�� y�� z���������С��������ݵ��ܶȣ�����λ��m�����źŵķ���Ƶ�ʣ���λ��MHz��

    if nargin == 0 %���������ʱ��Ĭ������
        room_x = 20;
        room_y = 15;
        room_z = 4;
        source_x = 10;
        source_y = 7.5;
        source_z = 1;
        grid_size = 0.1;
        f = 2400;
    end

    room_x = 1000 * room_x;
    room_y = 1000 * room_y;
    room_z = 1000 * room_z;
    source_x = 1000 * source_x;
    source_y = 1000 * source_y;
    source_z = 1000 * source_z;
    grid_size = 1000 * grid_size;
    
    %��糣���͵���ϵ�����ο����߸�����ص�����
    epsilon_c=10-1.2j;
    epsilon_w=6.1-1.2j;
    % epsilon_c=7.9-0.89j;
    % epsilon_w=6.2-0.69j;

    if room_x * room_y / grid_size^2 > 100000000
        disp('��ʾ������ʹ�ô����ͬʱ��������λ�õ��ϵ��ź�ǿ�ȣ�λ�õ����õĹ�����ܵ����ڴ治�㣬matlab���ܻῨ����');
        input('������س����˳���ctrl+c');
    end

    T = 1 / (f * 10^6);
    c = 3.0e8;
    lambda=c / (f * 10^6);

    %�õ��ռ������е������λ������,��z�̶������ź�Դһ���߶�
    [X, Y] = meshgrid(grid_size:grid_size:(room_x-grid_size), grid_size:grid_size:(room_y-grid_size));
    L = [X(:), Y(:)];
    L = [L, zeros(size(X(:))) + source_z];
    
    %% ֱ��·��
    %�������������λ�仯����糡E�ļ������档����ע����˵����λֻ�����ھ����������λ��
    d_direct = sqrt((L(:,1) - source_x).^2 + (L(:,2)-source_y).^2 + (L(:,3) - source_z).^2);%ÿ�������෢��Դ��ŷʽ����
    t_direct_0 = d_direct./1000./c;%ֱ��ʱ��
    p_direct = mod(t_direct_0*2*pi/T,2*pi);%ֱ����λ
    E_direct = (lambda./(4.*pi.*d_direct./1000));%����������E��һ�������ǳ�ǿ��С�����ͳ�ǿ������
    E0=E_direct.* exp(1i.*(-p_direct));

    %%
    %��������LΪ�������������ꡣÿ��Ϊһ�����ꡣ
    %�����Li��Ӧ��Ϊ�������������Ӧ�ľ����
    %��������鷴��·���ֱ����

    %% ǰƽ�淴��·����ǰ���������µ���˼�ǣ���վ������������У��泯y�ᣬ��ʱ��������ֱ��Ϊǰ���������£�
    Li=[L(:,1) , 2.*room_y-L(:,2) , L(:,3)]; %���㾵���
    d_reflect = sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2+(Li(:,3)-source_z).^2);%����·���ܳ���
    t_reflect_1 = d_reflect./1000./c;%ʱ��
    p_reflect = mod(t_reflect_1*2*pi/T,2*pi);%��λ
    thet = abs(atan((Li(:,2)-source_y)./(Li(:,1)-source_x)));%�����
    reflect_coefficient = (sin(thet)-sqrt(epsilon_w-(cos(thet)).^2))./(sin(thet)+sqrt(epsilon_w-(cos(thet)).^2));%����ϵ��Ҳ�Ǿ���
    E_reflect = (lambda./(4.*pi.*d_reflect./1000)) .*  reflect_coefficient;
    E1=E_reflect .* exp(1i.*(-p_reflect));%���ӳٵ���λ�ӽ������뷴����ɵ�˥������λ�仯����һ��

    %% ��ƽ�淴��·��
    Li=[L(:,1) , -L(:,2) , L(:,3)]; %���㾵���
    d_reflect = sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2+(Li(:,3)-source_z).^2);%����·���ܳ���
    t_reflect_2 = d_reflect./1000./c;%ʱ��
    p_reflect = mod(t_reflect_2*2*pi/T,2*pi);%��λ
    thet = abs(atan((Li(:,2)-source_y)./(Li(:,1)-source_x)));%�����
    reflect_coefficient = (sin(thet)-sqrt(epsilon_w-(cos(thet)).^2))./(sin(thet)+sqrt(epsilon_w-(cos(thet)).^2));
    E_reflect = (lambda./(4.*pi.*d_reflect./1000)) .*  reflect_coefficient;
    E2=E_reflect .* exp(1i.*(-p_reflect));%���ӳٵ���λ�ӽ������뷴����ɵ�˥������λ�仯����һ��

    %% ��ƽ�淴��·��
    Li=[-L(:,1) , L(:,2) , L(:,3)]; %���㾵���
    d_reflect = sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2+(Li(:,3)-source_z).^2);%����·���ܳ���
    t_reflect_3 = d_reflect./1000./c;%ʱ��
    p_reflect = mod(t_reflect_3*2*pi/T,2*pi);%��λ
    thet = abs(atan((Li(:,1)-source_x)./(Li(:,2)-source_y)));%�����
    reflect_coefficient = (sin(thet)-sqrt(epsilon_w-(cos(thet)).^2))./(sin(thet)+sqrt(epsilon_w-(cos(thet)).^2));
    E_reflect = (lambda./(4.*pi.*d_reflect./1000)) .*  reflect_coefficient;
    E3=E_reflect .* exp(1i.*(-p_reflect));%���ӳٵ���λ�ӽ������뷴����ɵ�˥������λ�仯����һ��

    %% ��ƽ�淴��·��
    Li=[2*room_x-L(:,1) , L(:,2) , L(:,3)]; %���㾵���
    d_reflect = sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2+(Li(:,3)-source_z).^2);%����·���ܳ���
    t_reflect_4 = d_reflect./1000./c;%ʱ��
    p_reflect = mod(t_reflect_4*2*pi/T,2*pi);%��λ
    thet = abs(atan((Li(:,1)-source_x)./(Li(:,2)-source_y)));%�����
    reflect_coefficient = (sin(thet)-sqrt(epsilon_w-(cos(thet)).^2))./(sin(thet)+sqrt(epsilon_w-(cos(thet)).^2));
    E_reflect = (lambda./(4.*pi.*d_reflect./1000)) .*  reflect_coefficient;
    E4=E_reflect .* exp(1i.*(-p_reflect));%���ӳٵ���λ�ӽ������뷴����ɵ�˥������λ�仯����һ��

    %% ��ƽ�淴��·��
    %%%2014.12.5������ƽ��ķ���·�������޸ģ���ֱ������б�������ʱ�򰡣����ڷ���ͼ��С��һЩ��Ȼ���ڷֽ�Ϊ��ֱ����ĵ糡��Ҫ��С��
    Li=[L(:,1) , L(:,2) , 2*room_z-L(:,3)]; %���㾵���
    d_reflect = sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2+(Li(:,3)-source_z).^2);%����·���ܳ���
    t_reflect_5 = d_reflect./1000./c;%ʱ��
    p_reflect = mod(t_reflect_5*2*pi/T,2*pi);%��λ
    thet = abs(atan((Li(:,3)-source_z)./sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2)));%�����
    reflect_coefficient = (-sin(thet).*epsilon_c+sqrt(epsilon_c-(cos(thet)).^2))./(epsilon_c.*sin(thet)+sqrt(epsilon_c-(cos(thet)).^2));%���ڵķ���ϵ��Ҳ�Ǿ�����
    E_reflect = (lambda./(4.*pi.*d_reflect./1000)) .*  reflect_coefficient;
    E5=E_reflect .* exp(1i.*(-p_reflect));%���ӳٵ���λ�ӽ������뷴����ɵ�˥������λ�仯����һ��
    E5=E5  .*   cos(pi*sin(thet)/2)./(cos(thet)+0.00001)  .*  cos(thet); %����ƽ��ĵ糡���ڷ���ͼ�Լ���ֱ�ֽ���Ҫ�������
    %% ��ƽ�淴��·��
    Li=[L(:,1) , L(:,2) , -L(:,3)]; %���㾵���
    d_reflect = sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2+(Li(:,3)-source_z).^2);%����·���ܳ���
    t_reflect_6 = d_reflect./1000./c;%ʱ��
    p_reflect = mod(t_reflect_6*2*pi/T,2*pi);%��λ
    thet = abs(atan((Li(:,3)-source_z)./sqrt((Li(:,1)-source_x).^2+(Li(:,2)-source_y).^2)));%�����
    reflect_coefficient = (-sin(thet).*epsilon_c+sqrt(epsilon_c-(cos(thet)).^2))./(epsilon_c.*sin(thet)+sqrt(epsilon_c-(cos(thet)).^2));
    E_reflect = (lambda./(4.*pi.*d_reflect./1000)) .*  reflect_coefficient;
    E6=E_reflect .* exp(1i.*(-p_reflect));%���ӳٵ���λ�ӽ������뷴����ɵ�˥������λ�仯����һ��
    E6=E6  .*   cos(pi*sin(thet)/2)./(cos(thet)+0.00001)  .*  cos(thet); %����ƽ��ĵ糡���ڷ���ͼ�Լ���ֱ�ֽ���Ҫ�������
    
    
    error1 = 0.05*rand(1,1);%����ÿ���źŵ��������
    error2 = 0.05*rand(1,1);
    error3 = 0.05*rand(1,1);
    error4 = 0.05*rand(1,1);
    error5 = 0.05*rand(1,1);
    error6 = 0.05*rand(1,1);
    E = E0 + E1*(1-error1) + E2*(1-error2)  + E3*(1-error3) + E4*(1-error4) + E5*(1-error5) + E6*(1-error6);%�������еĵ糡ǿ�ȴ��������·���ͷ������ı������ٳ���˥��ϵ��
    Power_all = 20 * log10(abs(E)) + 2 * 2.15;%�ϳɵĹ��ʡ�ʵ������һ��˥��ϵ����˳���ڼ��������������棬����Ϊ2.15dbi
    Power_all = reshape(Power_all, room_y / grid_size - 1, room_x / grid_size - 1)';
end
