// rand_val = curand_uniform(curand_states[idx]);
<% dependent_vars_count = len(list(sde.row_iterator('type', ['dependent variable']))) %>
${list(sde.row_iterator('derivative_order', [dependent_vars_count-1]))[0].Index}_next = ${list(sde.row_iterator('derivative_order', [dependent_vars_count-1]))[0].Index} + rhs * dt;
% for derivative_order in reversed(range(dependent_vars_count-1)):
    ${list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index}_next = ${list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index} + ${list(sde.row_iterator('derivative_order', [derivative_order+1]))[0].Index} * dt;
% endfor
rhs_next = ${sde.rhs_string[-1]};

% for independent_var in sde.row_iterator('type', ['independent variable']):
    ${independent_var.Index} += d${independent_var.Index};
% endfor

% for dependent_var in sde.row_iterator('type', ['dependent variable']):
    ${dependent_var.Index} = ${dependent_var.Index}_next;
% endfor
rhs = rhs_next;