function itermat = generate_all_onestep_deviating_policies(num_states,num_actions,polopt)
%GENERATE_ALL_DETERMINISTIC_POLICIES Summary of this function goes here
%   Detailed explanation goes herecols = 1;
itermat = repmat(polopt',num_states*(num_actions-1),1);
iter=0;
for state = 1:num_states
    for action = 1:num_actions
        if action ~= polopt(state)
            iter=iter+1;
            itermat(iter,state) = action;
        end
    end
end

fprintf("Total one step deviating policies\n");
disp(length(itermat));
end

