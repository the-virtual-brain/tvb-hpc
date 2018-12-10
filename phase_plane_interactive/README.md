# CodeJam 9 code generators

Here we explore the code synthesis without the loopy framework. The basic idea is to stick with the simple structured model representation and use a combination of AST transformations and active code templates to target various scenarios and architectures. Ideally, the templates will be derived from existing hand-written implementations.



Relevant files:
* `models.py` : example of model structured definition

* `dsl2python.py` : simple templated synthesis of a TVB-like Python class with a phase-plane interactive plot

  *  `template.py` active template for Python model class
  *  `ppi.py` : phase-plane interactive plot

* `dsl2C.py` : example of AST transformations for C-like languages

  *  `template.cu` active template for Cuda dfun function




## Model description

This was adapted from the loopy implementation. Note, that this is more a data-structure (string drifts), than a class with executable functions. Could be as well a JSON object... 

While it would be better to have  a functional model class, it is difficult to create one without introducing redundancy in the notation -- especially in the drift function definitions. So let's just stick with this: 

```Python
class G2DO:
    "Generic nonlinear 2-D (phase plane) oscillator."
    state = 'W', 'V'
    limit = (-5, 5), (-5, 5)
    input = 'c_0'
    param = 'a'
    const = {'tau': 1.0, 'I': 0.0, 'a': -2.0, 'b': -10.0, 'c': 0.0, 'd': 0.02,
             'e': 3.0, 'f': 1.0, 'g': 0.0, 'alpha': 1.0, 'beta': 1.0,
             'gamma': 1.0}
    drift = (
        'd * tau * (alpha*W - f*V**3 + e*V**2 + g*V + gamma*I + gamma*c_0)',
        'd * (a + b*V + c*V**2 - beta*W) / tau'
    )
    diffs = 1e-3, 1e-3
    obsrv = 'W', 'V'

```



## Code synthesis

For both C and Python we use the `mako` template engine to fill templates preparing the boilerplate for the `drift` expressions.

The advantage of the `drift` having Python syntax is, that we can use the built-in `ast` library for parsing, and the `ctree` library for basic transformations and code synthesis. For the demo, the simple operator conversion had to be extended to cope with `**`. 

In future, we can use the `ctree` framework for more complex transformations, such as array index arithmetic etc.



A demo Python code synthesized for the phase-plane interactive plot:

```Python
class Generic2D:

    def __init__(self,tau=1.0, I=0.0, a=-2.0, b=-10.0, c=0.0, d=0.02, e=3.0, f=1.0, g=0.0, alpha=1.0, beta=1.0, gamma=1.0):
        self.limit = ((-5, 5), (-5, 5))
        self.tau = tau
        self.I = I
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def dfun(self,state_variables, *args, **kwargs):
         
        W, V = state_variables

        tau = self.tau
        I = self.I
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f
        g = self.g
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        c_0 = 0.0
        
        dW = d * tau * (alpha*W - f*V**3 + e*V**2 + g*V + gamma*I + gamma*c_0)
        dV = d * (a + b*V + c*V**2 - beta*W) / tau
        return [W, V] 
```



A demo code synthesized to fit into an old code generator for TVB. 

```C
void model_dfun(
        float * _dx, float *_x, float *mmpr, float*input
    )   
{       
    float tau = mmpr[n_thr*0];
    float I = mmpr[n_thr*1];
    float a = mmpr[n_thr*2];
    float b = mmpr[n_thr*3];
    float c = mmpr[n_thr*4];
    float d = mmpr[n_thr*5];
    float e = mmpr[n_thr*6];
    float f = mmpr[n_thr*7];
    float g = mmpr[n_thr*8];
    float alpha = mmpr[n_thr*9];
    float beta = mmpr[n_thr*10];
    float gamma = mmpr[n_thr*11];
    
    float W = _x[n_thr*0];
    float V = _x[n_thr*1];

    float c_0 = input[0];

    _dx[n_thr*0] = d * tau * (alpha * W - f * pow(V, 3) + e * pow(V, 2) + g * V + gamma * I + gamma * c_0);
    _dx[n_thr*1] = d * tau * (alpha * W - f * pow(V, 3) + e * pow(V, 2) + g * V + gamma * I + gamma * c_0);

}

```



