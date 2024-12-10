import random
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []
    

class Neuron(Module):
    '''
    This class is used to define each Neuron and initialize them with weights and bias of the appropriate dimensions
    '''
    def __init__(self, nin, activation:str, nonlin = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.act = activation
        # used to indicate if we want the activation function to be used or not i.e. 
        # whether we want to add non-linearity or not
        self.nonlin = nonlin 

    def __call__(self, x):
        '''
        When the object will be called automatically the W*X + b computation will happen
        '''
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin:
            if self.act == 'relu':
                return act.relu()
            elif self.act == 'tanh':
                return act.tanh()
            elif self.act == 'sigmoid':
                return act.sigmoid()
        else:
            return act
    
    def parameters(self):
        '''
        Puts all the parameters (weights and Bias) in a single list so that during backpropagation 
        it can be easily accessed and updated
        '''
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self.act if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    

class Layer(Module):
    '''
    Creates a Layer of Neurons and the required connections given the number of input(nin) and output(nouts) neurons
    '''
    def __init__(self, nin, nouts, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nouts)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        num_out_nodes = [layer[0] for layer in nouts]
        activations = [layer[1] for layer in nouts]
        sz = [nin] + num_out_nodes
        self.layers = [Layer(sz[i], sz[i+1], 
                             activation=activations[i],
                             nonlin=i!=len(nouts)) for i in range(len(num_out_nodes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    

# # this the input example that will be passed to MLP when initialized.
# input_format = (2, [(16, 'relu'), (16, ('relu')), (1, 'sigmoid')]) 
# nin = 2
# nouts = [(16, 'relu'), (16, ('relu')), (1, 'sigmoid')]