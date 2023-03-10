# ----------------------------------------------------------------------------------------------------

class EWC:
    def __init__(self, model, weight) -> None:
        self.model = model
        self.weight = weight
    
    def update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(buff_param_name+'_estimated_mean', param.data.clone())
            
    def update_fisher_params(self, data_loader):
        buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(buff_param_names, model.parameters()):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.grad.data.clone() ** 2)
            
    def register_ewc_params(self, data_loader):
        self.update_fisher_params(data_loader)
        self.update_mean_params()

    def compute_consolidation_loss(self):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (self.weight / 2) * sum(losses)
        except AttributeError:
            return 0
        

# ----------------------------------------------------------------------------------------------------        


class L2Reg:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, model1, model2):
        
        l2loss = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            l2loss += self.alpha * torch.norm(p1 - p2)
            
        return l2loss

    def __repr__(self):
        return f"L2Reg({self.alpha})"
    
# ----------------------------------------------------------------------------------------------------