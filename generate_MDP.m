function [P,rewards,discount_factor] = generate_MDP(num_states,num_actions,reward_bounds)
%GENERATE_MDP: generate transition kernel, discount factor and reward based
%on inputs

discount_factor = rand; % U(0,0.5)
rewards = (reward_bounds(2) - reward_bounds(1))*rand(num_states,1) + reward_bounds(1); % U(reward_bounds(0),reward_bounds(1))

% generate transition kernel P(x',x,a), \sum_{x'} P(x',x,a) = 1 for all
% x,a.
P = rand(num_states,num_states,num_actions);
for action = 1:num_actions
    for state = 1:num_states
        P(state,:,action) = P(state,:,action)/sum(P(state,:,action));
    end
end

end

