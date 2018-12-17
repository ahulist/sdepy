#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

% if len(list(sde.row_iterator('type', 'noise'))) > 0:
    __device__ curandState_t* curand_states[${sde.settings['simulation']['number_of_threads']}];
% endif

% for sim_name, sim_value in sde.settings['constants'].items():
    __constant__ ${'float' if isinstance(sim_value, float) else 'int'} ${sim_name} = ${sim_value};
% endfor

% for row in sde.row_iterator('type', 'independent variable'):
    __constant__ ${'float' if isinstance(row.step, float) else 'int'} d${row.Index} = ${row.step};
% endfor

% for row in sde.row_iterator('type', 'parameter'):
    __device__ float parameter_${row.Index}_values[${len(row.values)}] = {\
    ${','.join([str(x) for x in row.values])}\
    };\
% endfor

__device__ int parameters_values_lens[${len(list(sde.row_iterator('type', 'parameter')))}] = {
    ${','.join([str(len(x.values)) for x in sde.row_iterator('type', 'parameter')])}
};

__device__ float* parameters_values[${len(list(sde.row_iterator('type', 'parameter')))}] = {
    ${','.join(['parameter_'+str(x.Index)+'_values' for x in sde.row_iterator('type', 'parameter')])}
};

__device__ float data [${sde.settings['simulation']['number_of_threads']}][${1 + len(list(sde.row_iterator('type', 'independent variable'))) + 3*len(list(sde.row_iterator('type', 'dependent variable')))}];

## #######################

__global__ void initkernel(int seed)
{
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    % if len(list(sde.row_iterator('type', 'noise'))) > 0:
        curandState_t* s = new curandState_t;
        curand_init(seed*idx, idx, 0, s); // czy mnozenie przez idx jest potrzebne?
        curand_states[idx] = s;
    % endif
}

<%
    def get_dep_indep_params_string(dep=True, indep=True, type_prefix=True, next_postfix=True):
        str_ = ' '
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
	__device__ __inline__ float ${list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index}_diff(int idx, ${get_dep_indep_params_string(dep=True, indep=True, type_prefix=True, next_postfix=False)})
	{
	    return ${sde.rhs_string[derivative_order]};
	}
% endfor

__device__ __inline__ void calc_avg(float &current_avg, float new_value, int current_step)
{
    current_avg += (new_value - current_avg) / (current_step % steps_per_period + 1);
}

extern "C" __global__ void prepare_simulation()
{
    //int idx =  blockIdx.x  *blockDim.x + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
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

extern "C" __global__ void continue_simulation()
{
    //int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    float my_params[${len(list(sde.row_iterator('type','parameter')))}];
    int index = (int) idx / ${sde.settings['simulation']['paths']};
    for(int i=0; i<${len(list(sde.row_iterator('type','parameter')))}; i++){
        my_params[i] = parameters_values[i][index%parameters_values_lens[i]];
        index = (int) index / parameters_values_lens[i];
    }
    
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
    
    <%include file="${sde.settings['simulation']['integration_method']}.mako" args="what_u_want='integration_initialization',sde=sde, dependent_vars_count=dependent_vars_count, derivative_order=derivative_order, get_dep_indep_params_string=get_dep_indep_params_string"/>
    
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
         <%include file="${sde.settings['simulation']['integration_method']}.mako" args="what_u_want='integration',sde=sde, dependent_vars_count=dependent_vars_count, derivative_order=derivative_order, get_dep_indep_params_string=get_dep_indep_params_string"/>
        
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

extern "C" __global__ void end_simulation(float *summary)
{
    //int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    <% len_ = len(list(sde.row_iterator('type', ['dependent variable']))) %>
    % for counter, row in enumerate(sde.row_iterator('type', ['dependent variable'])):
        summary[idx*${3*len_}+${counter*3+0}] = data[idx][${sde.lookup[str(row.Index)]}]; // ${row.Index}
        summary[idx*${3*len_}+${counter*3+1}] = data[idx][${sde.lookup['avg_period_'+str(row.Index)]}]; // avg_period_${row.Index}
        summary[idx*${3*len_}+${counter*3+2}] = data[idx][${sde.lookup['avg_periods_'+str(row.Index)]}]; // avg_periods_${row.Index}
    % endfor
}