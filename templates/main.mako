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

<% params_to_scan = sum([1 for row in sde.row_iterator('type', 'parameter') if len(row.values) > 1]) %>
% if params_to_scan > 0:
    % for row in sde.row_iterator('type', 'parameter'):
        % if len(row.values) > 1:
            __device__ float parameter_${row.Index}_values[${len(row.values)}] = {\
            ${','.join([str(x) for x in row.values])}\
            };\
        % endif
    % endfor

    __device__ int parameters_values_lens[${params_to_scan}] = {
        ${','.join([str(len(x.values)) for x in sde.row_iterator('type', 'parameter') if len(x.values) > 1])}
    };

    __device__ float* parameters_values[${params_to_scan}] = {
        ${','.join(['parameter_'+str(x.Index)+'_values' for x in sde.row_iterator('type', 'parameter') if len(x.values) > 1])}
    };
% endif

__device__ ${sde.settings['simulation']['precision']} data [${sde.settings['simulation']['number_of_threads']}][${1 + len(list(sde.row_iterator('type', 'independent variable'))) + 3*len(list(sde.row_iterator('type', 'dependent variable')))}];

## #######################

__global__ void initkernel(int seed)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    % if len(list(sde.row_iterator('type', 'noise'))) > 0:
    if(idx < ${sde.settings['simulation']['number_of_threads']}){
        curandState_t* s = new curandState_t;
        if (s != 0) {
            curand_init(seed*idx, idx, 0, s); // czy mnozenie przez idx jest potrzebne? ta linia rzuca LogicError: cuMemcpyHtoD failed: an illegal memory access was encountered
        }
        curand_states[idx] = s;
    }
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
	__device__ __inline__ float ${list(sde.row_iterator('derivative_order', [derivative_order]))[0].Index}_diff(int idx, ${get_dep_indep_params_string(dep=True, indep=True, type_prefix=True, next_postfix=False)}${''.join([', float '+str(row.Index) for row in sde.row_iterator('type', 'parameter') if len(row.values)>1])})
	{
	    return ${sde.rhs_string[derivative_order]};
	}
% endfor

extern "C" __global__ void prepare_simulation()
{
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
            data[idx][${free_index}] = 0.0f; // mean_${row.Index}
            <%
            sde.lookup['mean_' + str(row.Index)] = free_index
            free_index += 1
            %>\
            \
            data[idx][${free_index}] = 0.0f; // std_dev_${row.Index}
            <%
            sde.lookup['std_dev_' + str(row.Index)] = free_index
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
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    % if params_to_scan > 0:
        float my_params[${params_to_scan}];
        int index = (int) idx / ${sde.settings['simulation']['paths']};
        for(int i=0; i<${params_to_scan}; i++){
            my_params[i] = parameters_values[i][index%parameters_values_lens[i]];
            index = (int) index / parameters_values_lens[i];
        }
    % endif
    int current_step = (int) data[idx][${sde.lookup['current step']}];
    % for row in sde.row_iterator('type', ['dependent variable', 'independent variable']):
        float ${row.Index} = data[idx][${sde.lookup[str(row.Index)]}];
        % if row.type == 'dependent variable':
            float ${row.Index}_next;
            float ${row.Index}_diff_value;
            float mean_${row.Index} = data[idx][${sde.lookup['mean_'+str(row.Index)]}];
            float std_dev_${row.Index} = data[idx][${sde.lookup['std_dev_'+str(row.Index)]}];
        % endif
    % endfor
    float tmp_mean = 0;
    
    <%include file="${sde.settings['simulation']['integration_method']}.mako" args="what_u_want='integration_initialization',sde=sde, dependent_vars_count=dependent_vars_count, derivative_order=derivative_order, get_dep_indep_params_string=get_dep_indep_params_string"/>
    
    <% dependent_vars = len(list(sde.row_iterator('type', 'dependent variable'))) %>

    for (int i = 0; i < steps_per_kernel_call; i++) {
        /**
    	 * Integration
    	 */
         <%include file="${sde.settings['simulation']['integration_method']}.mako" args="what_u_want='integration',sde=sde, dependent_vars_count=dependent_vars_count, derivative_order=derivative_order, get_dep_indep_params_string=get_dep_indep_params_string"/>
        
        /**
         * Averaging
         */
        if(current_step >= transients){
        % for row in sde.row_iterator('type', 'dependent variable'):
            tmp_mean = mean_${row.Index} + (${row.Index} - mean_${row.Index})/(current_step + 1.0 - transients);
            std_dev_${row.Index} = std_dev_${row.Index} + (${row.Index} - mean_${row.Index})*(${row.Index} - tmp_mean);
            mean_${row.Index} = tmp_mean;
        % endfor
        }
        
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
            data[idx][${sde.lookup['mean_'+str(row.Index)]}] = mean_${row.Index};
            data[idx][${sde.lookup['std_dev_'+str(row.Index)]}] = std_dev_${row.Index};
        % endif
    % endfor
}

extern "C" __global__ void get_values(float *summary)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    <% len_ = len(list(sde.row_iterator('type', ['dependent variable']))) %>
    % for counter, row in enumerate(sde.row_iterator('type', ['dependent variable'])):
        summary[idx*${3*len_}+${counter*3+0}] = data[idx][${sde.lookup[str(row.Index)]}]; // ${row.Index}
        summary[idx*${3*len_}+${counter*3+1}] = data[idx][${sde.lookup['mean_'+str(row.Index)]}]; // mean_${row.Index}
        summary[idx*${3*len_}+${counter*3+2}] = data[idx][${sde.lookup['std_dev_'+str(row.Index)]}]; // std_dev_${row.Index}
    % endfor
}