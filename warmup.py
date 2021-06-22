class LinearWarmuper():
    def __init__(self, optimizer, steps, factor):
        self. optimizer = optimizer
        for param_group in optimizer.param_groups:
            self.initial_lr=param_group['lr']
        self.steps = steps
        self.current_step = 0
        self.factor = factor

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr*(1+(self.factor-1)*(self.current_step/self.steps))