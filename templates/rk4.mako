<%page args="what_u_want, sde, dependent_vars_count, derivative_order, get_dep_indep_params_string"/>\
% if what_u_want in ['integration_initialization', 'integration']:
    % if what_u_want == 'integration':
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
    % endif

    
    <%
        indeps = [x.Index for x in sde.row_iterator('type', 'independent variable')]
        deps = [x.Index for x in sde.row_iterator('type', 'dependent variable')]
    %>
    
    % for dep_var in deps:
        % if what_u_want == 'integration_initialization':
            float 
        % endif
        rk4_${dep_var}_diff_1 = ${dep_var}_diff(idx, ${get_dep_indep_params_string(dep=True, indep=True, type_prefix=False, next_postfix=False)}${''.join([', my_params[{}]'.format(x) for x in range(sum([1 for row in sde.row_iterator('type', 'parameter') if len(row.values)>1]))])});
    % endfor
    % for dep_var in deps:
        <%
        str_ = 'idx,'
        for indep in indeps:
            str_ += '{} + d{}/2.0, '.format(indep, indep)
            for dep in deps:
                str_ += '{} + d{}*rk4_{}_diff_1/2.0, '.format(dep, indep, dep)
        str_ = str_.strip()[:-1] + ''.join([', my_params[{}]'.format(x) for x in range(sum([1 for row in sde.row_iterator('type', 'parameter') if len(row.values)>1]))])
        %>\
        % if what_u_want == 'integration_initialization':
            float 
        % endif
        rk4_${dep_var}_diff_2 = ${dep_var}_diff(${str_});
    % endfor
    % for dep_var in deps:
        <%
        str_ = 'idx,'
        for indep in indeps:
            str_ += '{} + d{}/2.0, '.format(indep, indep)
            for dep in deps:
                str_ += '{} + d{}*rk4_{}_diff_2/2.0, '.format(dep, indep, dep)
        str_ = str_.strip()[:-1] + ''.join([', my_params[{}]'.format(x) for x in range(sum([1 for row in sde.row_iterator('type', 'parameter') if len(row.values)>1]))])
        %>\
        % if what_u_want == 'integration_initialization':
            float 
        % endif
        rk4_${dep_var}_diff_3 = ${dep_var}_diff(${str_});
    % endfor
    % for dep_var in deps:
        <%
        str_ = 'idx,'
        for indep in indeps:
            str_ += '{} + d{}, '.format(indep, indep)
            for dep in deps:
                str_ += '{} + d{}*rk4_{}_diff_3, '.format(dep, indep, dep)
        str_ = str_.strip()[:-1] + ''.join([', my_params[{}]'.format(x) for x in range(sum([1 for row in sde.row_iterator('type', 'parameter') if len(row.values)>1]))])
        %>\
        % if what_u_want == 'integration_initialization':
            float 
        % endif
        rk4_${dep_var}_diff_4 = ${dep_var}_diff(${str_});
    % endfor
    \
    % for dep_var in deps:
        ${dep_var}_diff_value = (rk4_${dep_var}_diff_1 + 2*rk4_${dep_var}_diff_2 + 2*rk4_${dep_var}_diff_3 + rk4_${dep_var}_diff_4)/6.0;
    % endfor
% endif