function A = generate_LP_constraints_onestep_lambda(lambda,P,P_polopt,R,d,num_states,num_actions,pol_opt)
%GENERATE_LP_CONSTRAINTS


% Equation (4) in Th. 3 of Alg. for IRL - Ng, Russell
A = zeros(num_states*(num_actions-1),num_states);
const_iter=0;

for state = 1:num_states
    for action = 1:num_actions
        if action ~= pol_opt(state)
            const_iter = const_iter + 1;
            A(const_iter,:) = P(state,:,pol_opt(state)) - P(state,:,action) ;
        end
    end
end

A = A/(eye(num_states) - d*P_polopt);

% generates perturbation values U(-lambda,lambda)
pert_mat = diag(2*lambda*rand(1,num_states) - lambda);

% B = vertcat(A + lambda*ones(length(A),1),A);
% A = vertcat(B,A- lambda*ones(length(A),1));

B = vertcat(A*(eye(num_states) + pert_mat),A);
A = vertcat(B,A*(eye(num_states) - pert_mat));

% sanity check - ensure A*R>= 0
if min(A*R)<0
    fprintf('Original reward fails feasibility test\n Terminating program');
    return
else
    fprintf('Original reward satisfies feasibility test\n');
end

end



