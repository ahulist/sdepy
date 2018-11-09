<%page args="what_u_want, sde, dependent_vars_count, derivative_order, get_dep_indep_params_string"/>\
% if what_u_want == 'integration':
    % for derivative_order in reversed(range(dependent_vars_count)):
        <% curr_var = list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index %>\
        ${curr_var}_next = ${curr_var} + ${curr_var}_diff(${get_dep_indep_params_string(dep=True, indep=True, type_prefix=False, next_postfix=False)}) * d${list(sde.row_iterator('type', ['independent variable']))[0].Index};
    % endfor

    % for independent_var in sde.row_iterator('type', ['independent variable']):
        ${independent_var.Index} += d${independent_var.Index};
    % endfor

    % for dependent_var in sde.row_iterator('type', ['dependent variable']):
        ${dependent_var.Index} = ${dependent_var.Index}_next;
    % endfor
% endif