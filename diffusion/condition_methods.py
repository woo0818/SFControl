from abc import ABC, abstractmethod
import torch
import pdb

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operators, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operators=operators, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operators, noiser, **kwargs):
        self.operators = operators
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def value(self, x_prev, x_0_hat, measurements, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            norm_diff = torch.tensor(0.0).to('cuda')
            collision_diff = torch.tensor(0.0).to('cuda')
            weight_type = measurements['weight_type']
                
            for key in self.operators.keys():
                operator = self.operators[key]
                
                weight = kwargs['model_kwargs']['y']['weight'][key]

                

                if weight_type[key] == 0:
                    measurement = measurements[key]
                    difference = measurement - operator.forward(x_0_hat, **kwargs)
                    norm_diff += weight * torch.linalg.norm(difference)
                elif weight_type[key] == 1:
                    collision_diff += weight * operator.forward(x_0_hat, **kwargs)

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()

        else:
            raise NotImplementedError
             
        return norm_diff, collision_diff
    
    def grad_and_value(self, x_prev, x_0_hat, measurements, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            weight_type = measurements['weight_type']
            norm_diff = 0.0
            for key in self.operators.keys():
                operator = self.operators[key]
                weight = kwargs['model_kwargs']['y']['weight'][key]
                if weight_type[key] == 0:
                    measurement = measurements[key]
                    difference = measurement - operator.forward(x_0_hat, **kwargs)
                    norm_diff += weight * torch.linalg.norm(difference)

                elif weight_type[key] == 1:
                    norm_diff += weight * operator.forward(x_0_hat, **kwargs)

            norm = torch.exp(-norm_diff)
            norm = torch.log(norm)

            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
        
    def grad_and_value_mc(self, x_prev, x_0_hat, measurements, log_var, n_mc, **kwargs):
        #n = 5
        sigma = torch.exp(0.5 * log_var)
        gamma = sigma/torch.sqrt(1+sigma**2)
        norm = 0.0
        differences = []
        if self.noiser.__name__ == 'gaussian':
            for _ in range(n_mc):
                noise = torch.randn_like(x_prev)
                x_0_hat += gamma * noise
                norm_diff = 0.0
                weight_type = measurements['weight_type']

                
                
                for key in self.operators.keys():
                    operator = self.operators[key]
                    weight = kwargs['model_kwargs']['y']['weight'][key]
                    if weight_type[key] == 0:
                        measurement = measurements[key]
                        difference = measurement - operator.forward(x_0_hat, **kwargs)
                        norm_diff += weight * torch.linalg.norm(difference)

                    elif weight_type[key] == 1:
                        norm_diff += weight * operator.forward(x_0_hat, **kwargs)

                norm += torch.exp(-norm_diff)
            
            norm = torch.log(norm/n_mc)
            #norm_grad = -torch.autograd.grad(outputs=norm, inputs=x_prev, create_graph=True)[0]

            #print("x_prev.grad_fn:", x_prev.grad_fn)

            #from torchviz import make_dot
            #make_dot(norm, params=dict(self.named_parameters())).render("graph", format="png")
            #print("x_0_hat.grad_fn:", x_0_hat.grad_fn)
            print(norm_diff.item()/weight)

            norm_grad = -torch.autograd.grad(
                outputs=norm,
                inputs=x_prev,
                #create_graph=True,  # 2차 미분 허용
                #retain_graph=True   # 그래프 유지
            )[0]
            #       pdb.set_trace()
            #print(norm_grad)
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operators, noiser, **kwargs):
        super().__init__(operators, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurements, log_var, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurements=measurements, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm

@register_conditioning_method(name='ps_mc')
class PosteriorSamplingMonteCarlo(ConditioningMethod):
    def __init__(self, operators, noiser, **kwargs):
        super().__init__(operators, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.n_mc_start = kwargs.get('n_mc', 5) #5 previous
        self.n_mc_final = 100
        self.steps = kwargs.get('steps', 1000)
        self.n_iter = 0
        #pdb.set_trace()

    def get_n_mc(self, iter):
        
        if iter <= self.steps - 10:
            return self.n_mc_start
        else:
            return self.n_mc_start
        #return self.n_mc

    def conditioning(self, x_prev, x_t, x_0_hat, measurements, log_var, **kwargs):
        self.n_iter += 1
        self.n_mc = self.get_n_mc(self.n_iter)
        norm_grad, norm = self.grad_and_value_mc(x_prev=x_prev, x_0_hat=x_0_hat, measurements=measurements, log_var=log_var, n_mc=self.n_mc, **kwargs)
        norm = self.value(x_prev=x_prev, x_0_hat=x_0_hat, measurements=measurements, **kwargs)
        x_t -= norm_grad * self.scale
        #print(self.n_iter)
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
