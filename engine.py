import math

class Value:   
    '''
    A class for creating our own data type that stores scalar values and it's gradients 
    '''
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    

    def __repr__(self):
        ''' represents/prints each value as follows '''
        return f"Value(data = {self.data}, grad = {self.grad})"
    

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    

    def __radd__(self, other):
        return self + other
    

    def __neg__(self):
        return self * -1
    

    def __sub__(self, other):
        return self + (-other)
    

    def __rsub__(self, other):
        return other + (-self)
    

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    

    def __rmul__(self, other):
        return self * other
    

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now."
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        
        out._backward = _backward
        return out
    

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    

    def __truediv__(self, other):
        return self * other**-1
    

    def __rtruediv__(self, other):
        return other * self**-1
    

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out
    

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    

    def sigmoid(self):
        x = self.data
        t = 1/(1+math.exp(-x))
        out = Value(t, (self, ), 'sigmoid')

        def _backward():
            self.grad += t*(1-t) * out.grad

        out._backward = _backward
        return out

    
    def backwards(self):
        '''
        Backpropagation Function that will 1st topologically sort all the parameters nodes and then compute gradients
        for each parameter one by one by calling the _backward method associated with that object.
        '''
        # Topological sort so that we can backpropagate only if the previous nodes computations are done.
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
