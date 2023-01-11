function A = generate_LP_constraints(P,P_polopt,R,d,num_states,num_actions,all_pols,pol_opt,num_constraints)
%GENERATE_LP_CONSTRAINTS

% Equation (4) in Th. 3 of Alg. for IRL - Ng, Russell
A = zeros( (num_constraints-1)*num_states,num_states);
const_iter=0;
for i = 1: num_constraints
    if (all_pols(i,:)') ~= pol_opt
        const_iter = const_iter + 1;
        P_pol = zeros(num_states,num_states);
        for state = 1:num_states
            P_pol(state,:) = P(state,:,all_pols(i,state));
        end
        A( (const_iter-1)*num_states + 1 : const_iter*num_states, :  ) = P_polopt - P_pol;    
    end
end

A = A/(eye(num_states) - d*P_polopt);

% sanity check - ensure A*R>= 0
if min(A*R)<0
    fprintf('Original reward fails feasibility test\n Terminating program');
    return
else
    fprintf('Original reward satisfies feasibility test\n');
end

end

