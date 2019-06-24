% converts data in csv to mat format
% author: Zhaonan Li zli@brandeis.edu
% created at: 2019-6-24

M = csvread('unconfined.csv');

% convert y to 1-y
M(:, 4) = 1 - M(:, 4);

pts_neg = struct('cdata', {});
pts_pos = struct('cdata', {});
pts_nuc = struct('cdata', {});
pts_pos_o = struct('cdata', {});

for c = 0:3
    for i = 11001:11800
        cur_class_mask = M(:, 2) == c;
        cur_frame_mask = M(:, 1) == i;
        cur_frame_data = M(cur_frame_mask & cur_class_mask, :);
        if ~isempty(cur_frame_data)
           cur_pos = [];
           for j = 1:size(cur_frame_data, 1)
               cur_pos(j, :) = cur_frame_data(j, 3:4);
           end
        end
        if c == 0
            pts_pos(i).cdata = cur_pos;
        end
        if c == 1
            pts_neg(i).cdata = cur_pos;
        end
        if c == 2
            pts_pos_o(i).cdata = cur_pos;
        end
        if c == 3
            pts_nuc(i).cdata = cur_pos;
        end
    end
end