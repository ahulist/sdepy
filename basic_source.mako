<%def name="dependentVars(print_type=True, print_next=False)">
    <%def name="nameNext(name, print_next=print_next)">
        % if print_next:
            ${name}_next
        % else:
            ${name}
        % endif
    </%def>
    % for var_order, var in variables.items():
        % if var_order >= 0:
            % if print_type:
                float \
            % endif
            ${nameNext(var.name)},
        % endif
    % endfor
    % if print_type:
        float \
    % endif
    ${nameNext('lhs')}
</%def>

<%
class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
%>
#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

## #######################
const int threads_total = ${sim['number_of_threads']};
__device__ curandState_t* curand_states[threads_total];

<%doc>
% for rhs_name, rhs_value in params.items():
__constant__ ${'float' if isinstance(rhs_value, float) else 'int'} rhs_${rhs_name} = ${rhs_value};
% endfor
</%doc>

% for sim_name, sim_value in sim.items():
__constant__ ${'float' if isinstance(sim_value, float) else 'int'} ${sim_name} = ${sim_value};
% endfor

% for var_order, var in variables.items():
    % if var_order == -1:
__constant__ float d${var.name} = ${var.step};
    % endif
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
    
    <% data_lookup = bidict() %>
    
    data[idx][0] = 0.0f;  // current step  <% data_lookup['current_step'] = 0 %>
    <% free_index = 1 %>
    % for var_order, var in sorted(variables.items()):
        float ${var.name} = data[idx][${free_index}] = ${var.value}; // ${var.name}  <% data_lookup[var.name] = free_index %><% free_index += 1 %>
        % if var_order >= 0:
            data[idx][${free_index}] = 0.0f; // avg_period_${var.name}  <% data_lookup['avg_period_'+var.name] = free_index %><% free_index += 1 %>
            data[idx][${free_index}] = 0.0f; // avg_periods_${var.name} <% data_lookup['avg_periods_'+var.name] = free_index %><% free_index += 1 %>
        % endif
    % endfor
    data[idx][${free_index}] = ${rhs}; // lhs  <% data_lookup['lhs'] = free_index %><% free_index += 1 %>
}

__device__ void afterstep(${dependentVars()})
{
    
}

extern "C" __global__ void continue_simulation(float *summary, float* output)
{
    int idx = threadIdx.x + threadIdx.y * 2;
    
    int current_step = (int) data[idx][${data_lookup['current_step']}];
    % for var_order, var in sorted(variables.items()):
        float ${var.name} = data[idx][${data_lookup[var.name]}];
        % if var_order >= 0:
            float avg_period_${var.name} = data[idx][${data_lookup['avg_period_'+var.name]}];
            float avg_periods_${var.name} = data[idx][${data_lookup['avg_periods_'+var.name]}];
            float ${var.name}_next;
        % endif
    % endfor
    float lhs = data[idx][${data_lookup['lhs']}];
    float lhs_next;

    for (int i = 0; i < steps_per_kernel_call; i++) {
        velocity_next = velocity + lhs * dt;
        position_next = position + velocity * dt;
        lhs_next = ${rhs};
##        acceleration_next = 1.0 / rhs_m * (-rhs_gamma * velocity_next - U_d(position_next) + rhs_a * cosf(rhs_omega * t) + rhs_f + rhs_D * rhs_xi);
        // rand_val = curand_uniform(curand_states[idx]);

        t += dt;

        position = position_next;
        velocity = velocity_next;
        lhs = lhs_next;
        
        // iterative mean https://stackoverflow.com/a/1934266/1185254
        calc_avg(avg_period_position, position, current_step);
        calc_avg(avg_period_velocity, velocity, current_step);
        
        if(current_step % steps_per_period == 0){
            calc_avg(avg_periods_position, avg_period_position, current_step/steps_per_period);
            calc_avg(avg_periods_velocity, avg_period_velocity, current_step/steps_per_period);
            avg_period_position = 0.0f;
            avg_period_velocity = 0.0f;
        }
        
        if(current_step % ${sim['afterstep_every']} == 0){
            afterstep(${dependentVars(False, True)});
        }
        
        current_step += 1;
    }
    
    data[idx][${data_lookup['current_step']}] = current_step;
    % for var_order, var in sorted(variables.items()):
        data[idx][${data_lookup[var.name]}] = ${var.name};
        % if var_order >= 0:
            data[idx][${data_lookup['avg_period_'+var.name]}] = avg_period_${var.name};
            data[idx][${data_lookup['avg_periods_'+var.name]}] = avg_periods_${var.name};
        % endif
    % endfor
    data[idx][${data_lookup['lhs']}] = lhs;
}

extern "C" __global__ void end_simulation(float *summary, float* output)
{
    int idx = threadIdx.x + threadIdx.y * 2;
    
    <% count = 0 %>
    % for var_order, var in sorted(variables.items()):
        % if var_order >= 0:
            summary[idx*6+${count}] = data[idx][${data_lookup[var.name]}]; // ${var.name} <% count += 1 %>
            summary[idx*6+${count}] = data[idx][${data_lookup['avg_period_'+var.name]}]; // avg_period_${var.name} <% count += 1 %>
            summary[idx*6+${count}] = data[idx][${data_lookup['avg_periods_'+var.name]}]; // avg_periods_${var.name} <% count += 1 %>
        % endif
    % endfor
}