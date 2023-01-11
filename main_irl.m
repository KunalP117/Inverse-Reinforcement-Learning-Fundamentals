clear;
clc;
close all;

% generate system parameters for MDP
num_states = 3; % Keep this fixed for this task
num_actions = 6;
reward_bounds = [0,10];

% generates transition kernel, reward vector (func. of state) and discount
% factor
[P,R,d] = generate_MDP(num_states,num_actions,reward_bounds);

R_mag = norm(R,2);

% Obtain optimal policy from policy iteration
[pol_opt,P_polopt] = policy_iteration(P,R,d,num_states,num_actions);


%%%%%% IRL to recover R %%%%%%%%%%%

% generate matrix that computes all possible deterministic policies
all_pols = generate_all_deterministic_policies(num_states,num_actions);
num_constraints = length(all_pols); % should be num_actions^num_states

all_onestepdeviating_pols = generate_all_onestep_deviating_policies(num_states,num_actions,pol_opt);
num_constraints_onestep = length(all_onestepdeviating_pols); % should be (num_actions-1)*num_states


lambda = 0.003;
Constraint_mat_1 = generate_LP_constraints(P,P_polopt,R,d,num_states,num_actions,all_pols,pol_opt,num_constraints);
Constraint_mat_2 = generate_LP_constraints_valuecomp(P,P_polopt,R,d,num_states,all_pols,pol_opt,num_constraints);
Constraint_mat_3 = generate_LP_constraints_onestep(P,P_polopt,R,d,num_states,num_actions,pol_opt);
Constraint_mat_4 = generate_LP_constraints_onestep_lambda(lambda,P,P_polopt,R,d,num_states,num_actions,pol_opt);

% lambda = 0.0075;
% Constraint_mat_2 = generate_LP_constraints_lambda(lambda,P,P_polopt,R,d,num_states,all_pols,pol_opt,num_constraints);

MAX_SIM_ITER = 10^3;
normpert = 0.5; % uncertainty in knowledge of reward norm

Feasible_set_1 = zeros(num_states,MAX_SIM_ITER);
for sim_iter = 1:MAX_SIM_ITER
    while 1
        point = (reward_bounds(2)-reward_bounds(1))*rand(num_states,1) + reward_bounds(1);
        point = point*R_mag*(1+normpert*rand)/norm(point,2);
        if min(Constraint_mat_1*point) >= 0
            Feasible_set_1(:,sim_iter) = point;
            break;
        end
    end
end

Feasible_set_2 = zeros(num_states,MAX_SIM_ITER);
for sim_iter = 1:MAX_SIM_ITER
    while 1
        point = (reward_bounds(2)-reward_bounds(1))*rand(num_states,1) + reward_bounds(1);
        point = point*R_mag*(1+normpert*rand)/norm(point,2);
        if min(Constraint_mat_2*point) >= 0
            Feasible_set_2(:,sim_iter) = point;
            break;
        end
    end
end

Feasible_set_3 = zeros(num_states,MAX_SIM_ITER);
for sim_iter = 1:MAX_SIM_ITER
    while 1
        point = (reward_bounds(2)-reward_bounds(1))*rand(num_states,1) + reward_bounds(1);
        point = point*R_mag*(1+normpert*rand)/norm(point,2);
        if min(Constraint_mat_3*point) >= 0
            Feasible_set_3(:,sim_iter) = point;
            break;
        end
    end
end

Feasible_set_4 = zeros(num_states,MAX_SIM_ITER);
for sim_iter = 1:MAX_SIM_ITER
    while 1
        point = (reward_bounds(2)-reward_bounds(1))*rand(num_states,1) + reward_bounds(1);
        point = point*R_mag*(1+normpert*rand)/norm(point,2);
        if min(Constraint_mat_4*point) >= 0
            Feasible_set_4(:,sim_iter) = point;
            break;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if num_states == 3
    % Plot feasible sets for both sets of equations separately
    figure();
    size = 10;
    size_truer = 200;
    color1 = 'red';
    color2 = 'blue';
    color3 = 'magenta';
    color4 = 'green';

    subplot(2,2,1);
    scatter3(Feasible_set_1(1,:),Feasible_set_1(2,:),Feasible_set_1(3,:),size,color1,'filled');
    hold on; scatter3(R(1),R(2),R(3),size_truer,"black",'filled');
    xlabel('R(1)');
    ylabel('R(2)');
    zlabel('R(3)');

    subplot(2,2,2);
    scatter3(Feasible_set_2(1,:),Feasible_set_2(2,:),Feasible_set_2(3,:),size,color2,'filled');
    hold on; scatter3(R(1),R(2),R(3),size_truer,"black",'filled');
    xlabel('R(1)');
    ylabel('R(2)');
    zlabel('R(3)');
    
    subplot(2,2,3);
    scatter3(Feasible_set_3(1,:),Feasible_set_3(2,:),Feasible_set_3(3,:),size,color3,'filled');
    hold on; scatter3(R(1),R(2),R(3),size_truer,"black",'filled');
    xlabel('R(1)');
    ylabel('R(2)');
    zlabel('R(3)');
    
    subplot(2,2,4);
    scatter3(Feasible_set_4(1,:),Feasible_set_4(2,:),Feasible_set_4(3,:),size,color4,'filled');
    hold on; scatter3(R(1),R(2),R(3),size_truer,"black",'filled');
    xlabel('R(1)');
    ylabel('R(2)');
    zlabel('R(3)');
end


% point in feasible set 1 (pass vanilla constraints) but fail constraint
% set 3
feasible_val_3_1 = min(Constraint_mat_3*Feasible_set_1);

% point in feasible set 1 (pass vanilla constraints) but fail constraint
% set 4
feasible_val_4_1 = min(Constraint_mat_4*Feasible_set_1);

fprintf("Fraction of points that pass Ng's inequalities but fail one step deviation:\n");
disp(sum( feasible_val_3_1 < 0)/length(feasible_val_3_1));

fprintf("Fraction of points that pass Ng's inequalities but fail regularized one-step inequalities:\n");
disp(sum( feasible_val_4_1 < 0)/length(feasible_val_4_1));



