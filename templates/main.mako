#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

const int threads_total = ${sde.settings['constants']['number_of_threads']};
__device__ curandState_t* curand_states[threads_total];

% for sim_name, sim_value in sde.settings['constants'].items():
    __constant__ ${'float' if isinstance(sim_value, float) else 'int'} ${sim_name} = ${sim_value};
% endfor

% for row in sde.row_iterator('type', 'independent variable'):
    __constant__ ${'float' if isinstance(row.step, float) else 'int'} d${row.Index} = ${row.step};
% endfor

__shared__ float data [4][12];

## #######################

__global__ void initkernel(int seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, idx, 0, s);
    curand_states[idx] = s;
}

<%
    def get_dep_indep_params_string(dep=True, indep=True, type_prefix=True, next_postfix=True):
        str_ = ''
        if indep:
            for variable in sde.row_iterator('type', ['independent variable']):
                if type_prefix:
                    str_ += 'float '
                str_ += variable.Index
                if next_postfix:
                    str_ += '_next'
                str_ += ','
        if dep:
            for variable in sde.row_iterator('type', ['dependent variable']):
                if type_prefix:
                    str_ += 'float '
                str_ += variable.Index
                if next_postfix:
                    str_ += '_next'
                str_ += ','
        return str_[:-1] 
    dependent_vars_count = len(list(sde.row_iterator('type', ['dependent variable'])))
%>
% for derivative_order in reversed(range(dependent_vars_count)):
	__device__ __inline__ float ${list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index}_diff(${get_dep_indep_params_string(dep=True, indep=True, type_prefix=True, next_postfix=False)})
	{
	    return ${sde.rhs_string[derivative_order]};
	}
% endfor

__device__ __inline__ void calc_avg(float &current_avg, float new_value, int current_step)
{
    current_avg += (new_value - current_avg) / (current_step % steps_per_period + 1);
}

extern "C" __global__ void prepare_simulation(float *summary, float* output)
{
    int idx = threadIdx.x + threadIdx.y * 2;
    
    <% free_index = 0 %>
    
    data[idx][${free_index}] = 0.0f;  // current step
    <% 
    sde.lookup['current step'] = free_index
    free_index += 1
    %>
    
    % for row in sde.row_iterator('type', ['dependent variable', 'independent variable']):
        float ${row.Index} = data[idx][${free_index}] = ${row.values}; // ${row.Index}
        <%
        sde.lookup[str(row.Index)] = free_index
        free_index += 1
        %>\
        \
        % if row.type == 'dependent variable':
            data[idx][${free_index}] = 0.0f; // avg_period_${row.Index}
            <%
            sde.lookup['avg_period_' + str(row.Index)] = free_index
            free_index += 1
            %>\
            \
            data[idx][${free_index}] = 0.0f; // avg_periods_${row.Index}
            <%
            sde.lookup['avg_periods_' + str(row.Index)] = free_index
            free_index += 1
            %>\
        % endif
    % endfor
}

__device__ void afterstep(${get_dep_indep_params_string(dep=True, indep=True, type_prefix=True, next_postfix=False)})
{
    
}

extern "C" __global__ void continue_simulation(float *summary, float* output)
{
    int idx = threadIdx.x + threadIdx.y * 2;
    
    int current_step = (int) data[idx][${sde.lookup['current step']}];
    % for row in sde.row_iterator('type', ['dependent variable', 'independent variable']):
        float ${row.Index} = data[idx][${sde.lookup[str(row.Index)]}];
        % if row.type == 'dependent variable':
            float ${row.Index}_next;
            float ${row.Index}_diff_value;
            float avg_period_${row.Index} = data[idx][${sde.lookup['avg_period_'+str(row.Index)]}];
            float avg_periods_${row.Index} = data[idx][${sde.lookup['avg_periods_'+str(row.Index)]}];
        % endif
    % endfor
    
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
    
    <% dependent_vars = len(list(sde.row_iterator('type', 'dependent variable'))) %>

    for (int i = 0; i < steps_per_kernel_call; i++) {
        /**
         * Averaging
         */
        // iterative mean https://stackoverflow.com/a/1934266/1185254
        % for row in sde.row_iterator('type', 'dependent variable'):
            calc_avg(avg_period_${row.Index}, ${row.Index}, current_step);
        % endfor
        
        if(current_step % (steps_per_period-1) == 0){
        	% for row in sde.row_iterator('type', 'dependent variable'):
        		calc_avg(avg_periods_${row.Index}, avg_period_${row.Index}, current_step/steps_per_period);
        		avg_period_${row.Index} = 0.0f;
        	% endfor
        }
        
        /**
    	 * Integration
    	 */ 
        % if sde.settings['simulation']['integration_method'] == 'euler':
##		    <%include file="euler.mako"/>
            % for derivative_order in reversed(range(dependent_vars_count)):
                <% curr_var = list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index %>\
                ${curr_var}_diff_value = ${curr_var}_diff(${get_dep_indep_params_string(dep=True, indep=True, type_prefix=False, next_postfix=False)});
            % endfor
		% elif sde.settings['simulation']['integration_method'] == 'rk4':
##			<%include file="rk4.mako"/>
            % for derivative_order in reversed(range(dependent_vars_count)):
                <% curr_var = list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index %>\
                ${curr_var}_diff_value = rk4_${curr_var}_diff;
            % endfor
		% endif
         
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
        
        /**
    	 * Afterstep
    	 */
        if(current_step % ${sde.settings['constants']['afterstep_every']} == 0){
            afterstep(${get_dep_indep_params_string(dep=True, indep=True, type_prefix=False, next_postfix=False)});
        }
        
        current_step += 1;
    }

    data[idx][${sde.lookup['current step']}] = current_step;
    % for row in sde.row_iterator('type', ['dependent variable', 'independent variable']):
        data[idx][${sde.lookup[str(row.Index)]}] = ${row.Index};
        % if row.type == 'dependent variable':
            data[idx][${sde.lookup['avg_period_'+str(row.Index)]}] = avg_period_${row.Index};
            data[idx][${sde.lookup['avg_periods_'+str(row.Index)]}] = avg_periods_${row.Index};
        % endif
    % endfor
}

extern "C" __global__ void end_simulation(float *summary, float* output)
{
    int idx = threadIdx.x + threadIdx.y * 2;
    
    <% len_ = len(list(sde.row_iterator('type', ['dependent variable']))) %>
    % for counter, row in enumerate(sde.row_iterator('type', ['dependent variable'])):
        summary[idx*${3*len_}+${counter*3+0}] = data[idx][${sde.lookup[str(row.Index)]}]; // ${row.Index}
        summary[idx*${3*len_}+${counter*3+1}] = data[idx][${sde.lookup['avg_period_'+str(row.Index)]}]; // avg_period_${row.Index}
        summary[idx*${3*len_}+${counter*3+2}] = data[idx][${sde.lookup['avg_periods_'+str(row.Index)]}]; // avg_periods_${row.Index}
    % endfor
}