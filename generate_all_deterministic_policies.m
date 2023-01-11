function itermat = generate_all_deterministic_policies(num_states,num_actions)
%GENERATE_ALL_DETERMINISTIC_POLICIES Summary of this function goes here
%   Detailed explanation goes herecols = 1;
itermat = (1:num_actions)';
cols = 1;
for state = 2:num_states
    cols = cols+1;
    itermat_big = zeros(length(itermat)*num_actions,cols);
    for action = 1:num_actions
        itermat_big( (action-1)*length(itermat)+1:action*length(itermat),:) = horzcat(itermat,action*ones(length(itermat),1));
    end
    itermat = itermat_big;
end

fprintf("Total policies\n");
disp(length(itermat));
end

