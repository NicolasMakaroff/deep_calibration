
import math
import sys
import torch

twopi = math.pi*2.0
halfpi = math.pi/2.0

def IsComplex(obj):
    return hasattr(obj, 're') and hasattr(obj, 'im')

def ToComplex(obj):
    if IsComplex(obj):
        return obj
    elif isinstance(obj, tuple):
        return Complex(*obj)
    else:
        return Complex(obj)

def PolarToComplex(r = 0, phi = 0, fullcircle = twopi):
    phi = phi * (twopi / fullcircle)
    return Complex(math.cos(phi)*r, math.sin(phi)*r)

def Re(obj):
    if IsComplex(obj):
        return obj.re
    return obj

def Im(obj):
    if IsComplex(obj):
        return obj.im
    return 0

class Complex:

    def __init__(self, re=0, im=0):
        _re = 0
        _im = 0
        if IsComplex(re):
            _re = re.re
            _im = re.im
        else:
            _re = re
        if IsComplex(im):
            _re = _re - im.im
            _im = _im + im.re
        else:
            _im = _im + im
        # this class is immutable, so setting self.re directly is
        # not possible.
        self.__dict__['re'] = _re
        self.__dict__['im'] = _im

    def __setattr__(self, name, value):
        raise TypeError('Complex numbers are immutable')

    def __hash__(self):
        if not self.im:
            return hash(self.re)
        return hash((self.re, self.im))

    def __repr__(self):
        if not self.im:
            return 'Complex(%r)' % (self.re,)
        else:
            return 'Complex(%r, %r)' % (self.re, self.im)

    def __str__(self):
        if not self.im:
            return repr(self.re)
        else:
            return 'Complex(%r, %r)' % (self.re, self.im)

    def __neg__(self):
        return Complex(-self.re, -self.im)

    def __pos__(self):
        return self

    def __abs__(self):
        return torch.sqrt(torch.pow(self.re,2) + torch.pow(self.im,2))

    def __int__(self):
        if self.im:
            raise ValueError("can't convert Complex with nonzero im to int")
        return int(self.re)

    def __long__(self):
        if self.im:
            raise ValueError("can't convert Complex with nonzero im to long")
        return long(self.re)

    def __float__(self):
        if self.im:
            raise ValueError("can't convert Complex with nonzero im to float")
        return float(self.re)

    def __cmp__(self, other):
        other = ToComplex(other)
        return cmp((self.re, self.im), (other.re, other.im))

    def __rcmp__(self, other):
        other = ToComplex(other)
        return cmp(other, self)

    def __nonzero__(self):
        return not (self.re == self.im == 0)

    abs = radius = __abs__

    def angle(self, fullcircle = twopi):
        return (fullcircle/twopi) * ((halfpi - torch.atan2(self.re, self.im)) % twopi)

    phi = angle

    def __add__(self, other):
        other = ToComplex(other)
        return Complex(self.re + other.re, self.im + other.im)

    __radd__ = __add__

    def __sub__(self, other):
        other = ToComplex(other)
        return Complex(self.re - other.re, self.im - other.im)

    def __rsub__(self, other):
        other = ToComplex(other)
        return other - self

    def __mul__(self, other):
        other = ToComplex(other)
        return Complex(self.re*other.re - self.im*other.im,
                       self.re*other.im + self.im*other.re)

    __rmul__ = __mul__

    def div(self, other):
        other = ToComplex(other)
        d = other.re*other.re + other.im*other.im
        return Complex((self.re*other.re + self.im*other.im) / d,
                       (self.im*other.re - self.re*other.im) / d)

    def __rdiv__(self, other):
        other = ToComplex(other)
        return other / self

    def __pow__(self, n, z=None):
        if z is not None:
            raise TypeError('Complex does not support ternary pow()')
        if IsComplex(n):
            if n.im:
                if self.im: raise TypeError('Complex to the Complex power')
                else: return torch.exp(torch.log(self.re)*n)
            n = n.re
        r = torch.pow(self.abs(), n)
        phi = n*self.angle()
        return Complex(torch.cos(phi)*r, torch.sin(phi)*r)

    def __rpow__(self, base):
        base = ToComplex(base)
        return torch.pow(base, self)

def Torchexp(z):
    r = torch.exp(z.re)
    return Complex(torch.cos(z.im)*r,torch.sin(z.im)*r)

def TorchCosh(z):
    result = Complex(torch.cosh(z.re)*torch.cos(z.im),torch.sinh(z.re)* torch.sin(z.im))
    return result

def TorchSinh(z):
    result = Complex(torch.sinh(z.re)*torch.cos(z.im) ,torch.cosh(z.re)*torch.sin(z.im))
    return result
    
    
    
    