import numpy as np
from sympy import sqrt, sin, symbols, Function
from IPython.display import display

# F = ma = sum of:
    # -friction [gamma]
    # noise [xi]
    # noise intensity [D]
    # potential(x) [U]
    # oscillating force amplitude [a]
    # oscillating force(omega, t)
    # constant force [f]

class Sde():
    
    @staticmethod
    def get_default_options():
        return {
            'simulation':{
                'spp':100,         # steps per period
                'periods':2,#000,    # number of periods in the simulation
                'paths':2,#56,       # number of paths to sample
                'samples':100,     # sample the position every N steps
                'transients_number':200,      # number of periods to ignore
                'transients_fraction':0.1,    # fraction of periods to ignore
                'transients_type':'fraction', # periods to ignore because of transients (fraction, number)
                'rng_seed':None,
                'precision':'single',         # precision of the floating-point numbers (single, double)
                'rng_generator':'kiss32', 
                'deterministic':False,        # do not generate any noises
            },
            'output':{
                'mode':'summary',  # output mode (summary, path)
                'format':'text',   # output file format (text, npy)
                'destination':'./sde_out'
            },
            'gpu':{
                'cuda':True,
            },
            'debug':{
                'enabled':True,
            }
        }
        
    def __init__(self, variables, parameters, functions, equation, options = get_default_options.__func__()):
        self.variables = variables
        self.parameters = parameters
        self.functions = functions
        self.equation = equation
        self.options = options
        
        self.name_to_sympy = {}
        for p in self.parameters:
            self.name_to_sympy[p.name] = p
        for v in self.variables:
            self.name_to_sympy[v.name] = v
        
        self._init_cuda()
    
    def _init_cuda(self):
        if self.options['gpu']['cuda']:
            import pycuda.driver as cuda
            cuda.init()

    
    def _validate(self):
        for symbol in self.equation.free_symbols:
            if symbol not in self.variables and symbol not in self.parameters:
                raise ValueError('Value(s) for \'%s\' not provided.' % symbol.name)
        
        # dt
        self.name_to_sympy['dt'] = symbols('dt')
        self.parameters[self.name_to_sympy['dt']] = 2*np.pi/self.parameters[self.name_to_sympy['omega']]
    
    def _simulate(self):
        if self.options['gpu']['cuda']:
            self._simulate_gpu()
        else:
            self._simulate_cpu()
            
    def _simulate_cpu(self):
        current_values = self.variables.copy()
        
        for path in range(self.options['simulation']['paths']):
            dt = self.parameters[self.name_to_sympy['dt']]
            
            for period in range(self.options['simulation']['periods']):
                for step in range(self.options['simulation']['spp']):
                    result = self.equation.evalf(subs = current_values)
                    current_values[self.name_to_sympy['t']] += dt
                    current_values[self.name_to_sympy['v']] += result * dt
                    current_values[self.name_to_sympy['x']] += current_values[self.name_to_sympy['v']] * dt
                    
    def _simulate_gpu(self):
        return
    
    def start(self):
        self._validate()
        self._simulate()
        

# variables with initial values
var_x, var_v, var_t = symbols('x v t')
variables = {
    var_x:0,
    var_v:1,
    var_t:0,
}

# parameters with values
par_gamma, par_amplitude, par_potential_aplitude, par_omega, par_const_force, par_noise_intensity \
    = symbols('gamma a Delta_U omega f D')
parameters = {
    par_gamma:1,
    par_amplitude:1,
    par_potential_aplitude:1,
    par_omega:1,
#     par_const_force:np.linspace(0, 1, 10),
    par_const_force:1,
    par_noise_intensity:1,
}

# functions
def oscillation():
    return sin

def potential():
    return sin

f_potential = Function('U')
f_oscillation = oscillation
f_noise = Function('xi')
functions = {f_potential, f_oscillation, f_noise}

# equation
equation = -par_gamma*var_v + par_potential_aplitude * f_potential(var_x) + par_amplitude * f_oscillation()(par_omega*var_t) \
    + par_const_force + sqrt(2*par_noise_intensity) * f_noise(var_t)
equation2 = -par_gamma*var_v + par_potential_aplitude * sin(var_x) + par_amplitude * sin(par_omega*var_t) \
    + par_const_force + sqrt(2*par_noise_intensity)

# runtime options
options = Sde.get_default_options()
options['gpu']['cuda'] = False
# SDE
SDE = Sde(variables, parameters, functions, equation2, options)
SDE.start()

from sympy import init_printing
init_printing()

display(equation)
display(equation.free_symbols)
print(f_potential)
print(variables)

display(equation.evalf())
display(equation.evalf(subs={par_const_force:1}))

print('\n--------------------\n')

equation2 = -par_gamma*var_v # + par_potential_aplitude * f_potential(var_x) + par_amplitude * f_oscillation(par_omega, var_t) + par_const_force + sqrt(2*par_noise_intensity) * f_noise(var_t)
display(equation2)
print(equation2.free_symbols)
display(equation2.evalf())
display(equation2.evalf(subs={'gamma':3}))

from sympy import Symbol, solve
aa = Symbol("a")
bb = Symbol("b")
cc = Symbol("c")
exp = (aa+bb)*40-(cc-aa)/0.5
print(exp)
print(exp.free_symbols)
print(solve(exp))
print(exp.evalf(subs={aa:0.2, 'b':0.05, 'c':0.7}))

