% for derivative_order in reversed(range(dependent_vars_count)):
    <% curr_var = list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index %>\
    ${curr_var}_diff_value = ${curr_var}_diff(${get_dep_indep_params_string(dep=True, indep=True, type_prefix=False, next_postfix=False)});
% endfor