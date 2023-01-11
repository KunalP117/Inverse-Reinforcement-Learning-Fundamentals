function [pol,P_pol] = policy_iteration(P,R,d,num_states,num_actions)
%POLICY_ITERATION
% Policy Iteration
pol = ones(num_states,1);
pol_next = pol;
V = zeros(num_states,1); %Value Function
while 1
    % compute value function
    P_pol = zeros(num_states,num_states);
    for state = 1:num_states
        P_pol(state,:) = P(state,:,pol(state));
    end
    V = (eye(num_states) - d*P_pol)\R; % V = (I - d P_{pol})^{-1}*R
    
    % improve policy
    for state = 1:num_states
        P_improv = reshape(P(state,:,:),[num_states,num_actions])';
        [~,pol_next(state)] = max(P_improv*V);
    end
    
    % terminate if pol_next = pol
    if pol == pol_next
        break;
    else
        pol = pol_next;
    end
end


% CHECK BELLMAN OPTIMALITY
fprintf("Optimal Deterministic Policy:\n");
disp(pol);
fprintf("\nMax. Bellman Error:\n");
disp(max(abs(V - (R + d*P_pol*V))));
end

