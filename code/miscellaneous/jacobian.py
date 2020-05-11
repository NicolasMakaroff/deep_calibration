import torch 
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

class Jacobian(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,inputs,outputs):
        assert inputs.requires_grad


        num_classes = outputs.size()[1]
        jacobian = torch.zeros(num_classes,*inputs.size())

        grad_output = torch.zeros(*outputs.size())
        if inputs.is_cuda :
            grad_output = grad_output.cuda()
            jacobian = jacobian.cuda()
        for i in range(num_classes):
            zero_gradients(inputs)

            grad_output.zero_()
            grad_output[:,i]=1
            outputs.backward(grad_output,retain_graph=True,create_graph=True)

            jacobian[i] = inputs.grad

        return jacobian