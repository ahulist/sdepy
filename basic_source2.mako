#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

const int threads_total = ${sde.settings['simulation']['number_of_threads']};
__device__ curandState_t* curand_states[threads_total];

% for sim_name, sim_value in sde.settings['simulation'].items():
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

__device__ void calc_avg(float &current_avg, float new_value, int current_step)
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
    
    data[idx][${free_index}] = ${sde.rhs_string[-1]}; // rhs
    <%
    sde.lookup['rhs'] = free_index
    free_index += 1
    %>
}

__device__ void afterstep(
    % for row in sde.row_iterator('type', ['dependent variable']):
        float ${row.Index},
    % endfor
    float rhs
)
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
            float avg_period_${row.Index} = data[idx][${sde.lookup['avg_period_'+str(row.Index)]}];
            float avg_periods_${row.Index} = data[idx][${sde.lookup['avg_periods_'+str(row.Index)]}];
        % endif
    % endfor
    float rhs = data[idx][${sde.lookup['rhs']}];
    float rhs_next;
    
    <% dependent_vars = len(list(sde.row_iterator('type', 'dependent variable'))) %>

    for (int i = 0; i < steps_per_kernel_call; i++) {
        // iterative mean https://stackoverflow.com/a/1934266/1185254
        calc_avg(avg_period_position, position, current_step);  <%doc>TODO: generowac!</%doc>
        calc_avg(avg_period_velocity, velocity, current_step);
        
        if(current_step % (steps_per_period-1) == 0){
        	% for row in sde.row_iterator('type', 'dependent variable'):
        		calc_avg(avg_periods_${row.Index}, avg_period_${row.Index}, current_step/steps_per_period);
        		avg_period_${row.Index} = 0.0f;
        	% endfor
        }
        
        <%include file="euler.mako"/>
        
        if(current_step % ${sde.settings['simulation']['afterstep_every']} == 0){
            afterstep(
            % for row in sde.row_iterator('type', ['dependent variable']):
                ${row.Index}_next,
            % endfor
            rhs_next
            );
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
    data[idx][${sde.lookup['rhs']}] = rhs;
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