<%page args="what_u_want, sde, dependent_vars_count, derivative_order, get_dep_indep_params_string"/>
% if what_u_want == 'integration_initialization':
    float rk4_v_diff_1 = v_diff(t, v, x);
	float rk4_x_diff_1 = x_diff(t, v, x);
	float rk4_v_diff_2 = v_diff(t + dt/2.0, v + dt*rk4_v_diff_1/2.0, x + dt*rk4_x_diff_1/2.0);
	float rk4_x_diff_2 = x_diff(t + dt/2.0, v + dt*rk4_v_diff_1/2.0, x + dt*rk4_x_diff_1/2.0);
	float rk4_v_diff_3 = v_diff(t + dt/2.0, v + dt*rk4_v_diff_2/2.0, x + dt*rk4_x_diff_2/2.0);
	float rk4_x_diff_3 = x_diff(t + dt/2.0, v + dt*rk4_v_diff_2/2.0, x + dt*rk4_x_diff_2/2.0);
	float rk4_v_diff_4 = v_diff(t + dt, v + dt*rk4_v_diff_3, x + dt*rk4_x_diff_3);
	float rk4_x_diff_4 = x_diff(t + dt, v + dt*rk4_v_diff_3, x + dt*rk4_x_diff_3);
	float rk4_v_diff = (rk4_v_diff_1 + 2*rk4_v_diff_2 + 2*rk4_v_diff_3 + rk4_v_diff_4)/6.0;
	float rk4_x_diff = (rk4_x_diff_1 + 2*rk4_x_diff_2 + 2*rk4_x_diff_3 + rk4_x_diff_4)/6.0;
% elif what_u_want == 'integration':
    % for derivative_order in reversed(range(dependent_vars_count)):
        <% curr_var = list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index %>\
        ${curr_var}_diff_value = rk4_${curr_var}_diff;
    % endfor

    % for derivative_order in reversed(range(dependent_vars_count)):
        <% curr_var = list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index %>\
        ${curr_var}_next = ${curr_var} + ${curr_var}_diff_value * d${list(sde.row_iterator('type', ['independent variable']))[0].Index};
    % endfor

    % for independent_var in sde.row_iterator('type', ['independent variable']):
        ${independent_var.Index} += d${independent_var.Index};
    % endfor

    % for dependent_var in sde.row_iterator('type', ['dependent variable']):
        ${dependent_var.Index} = ${dependent_var.Index}_next;
    % endfor

    rk4_v_diff_1 = v_diff(t, v, x);
	rk4_x_diff_1 = x_diff(t, v, x);
	rk4_v_diff_2 = v_diff(t + dt/2.0, v + dt*rk4_v_diff_1/2.0, x + dt*rk4_x_diff_1/2.0);
	rk4_x_diff_2 = x_diff(t + dt/2.0, v + dt*rk4_v_diff_1/2.0, x + dt*rk4_x_diff_1/2.0);
	rk4_v_diff_3 = v_diff(t + dt/2.0, v + dt*rk4_v_diff_2/2.0, x + dt*rk4_x_diff_2/2.0);
	rk4_x_diff_3 = x_diff(t + dt/2.0, v + dt*rk4_v_diff_2/2.0, x + dt*rk4_x_diff_2/2.0);
	rk4_v_diff_4 = v_diff(t + dt, v + dt*rk4_v_diff_3, x + dt*rk4_x_diff_3);
	rk4_x_diff_4 = x_diff(t + dt, v + dt*rk4_v_diff_3, x + dt*rk4_x_diff_3);
	rk4_v_diff = (rk4_v_diff_1 + 2*rk4_v_diff_2 + 2*rk4_v_diff_3 + rk4_v_diff_4)/6.0;
	rk4_x_diff = (rk4_x_diff_1 + 2*rk4_x_diff_2 + 2*rk4_x_diff_3 + rk4_x_diff_4)/6.0;
% endif