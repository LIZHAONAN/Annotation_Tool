% converts absolute positions in .mat file to relative positions (1-0)
% author: Zhaonan Li zli@brandeis.edu
% created at: 6-17-2019

path_to_old_mat = 'Saved_Results/6-12-RL.mat';
path_to_new_mat = 'Saved_Results/6-12-RL_relative.mat';
load(path_to_old_mat);

w = 736;
h = 748;

for i = 1:11800
    pts_pos(i).cdata = double(pts_pos(i).cdata);
    old_pos = pts_pos(i).cdata;
%     img = imread(strcat('Frames/images/', sprintf('%06d.jpg', i)));
%     [h,w,d] = size(img);
    if size(old_pos, 1) > 0
        old_pos(:, 1) = double(old_pos(:, 1)) / w;
        old_pos(:, 2) = double(old_pos(:, 2)) / h;
        pts_pos(i).cdata = old_pos;
    end
    
    pts_neg(i).cdata = double(pts_neg(i).cdata);
    old_neg = pts_neg(i).cdata;
    if size(old_neg, 1) > 0
        old_neg(:, 1) = double(old_neg(:, 1)) / w;
        old_neg(:, 2) = double(old_neg(:, 2)) / h;
        pts_neg(i).cdata = old_neg;
    end
    
    pts_pos_o(i).cdata = double(pts_pos_o(i).cdata);
    old_pos_o = pts_pos_o(i).cdata;
    if size(old_pos_o, 1) > 0
        old_pos_o(:, 1) = double(old_pos_o(:, 1)) / w;
        old_pos_o(:, 2) = double(old_pos_o(:, 2)) / h;
        pts_pos_o(i).cdata = old_pos_o;
    end
    
    pts_nuc(i).cdata = double(pts_nuc(i).cdata);
    old_nuc = pts_nuc(i).cdata;
    if size(old_nuc, 1) > 0
        old_nuc(:, 1) = double(old_nuc(:, 1)) / w;
        old_nuc(:, 2) = double(old_nuc(:, 2)) / h;
        pts_nuc(i).cdata = old_nuc;
    end
end

save(path_to_new_mat, 'k', 'pts_pos', 'pts_neg', 'pts_pos_o', 'pts_nuc');