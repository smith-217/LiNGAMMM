

def init_msgs_run(
    self
    ,refresh
    ,lambda_control = None
    ,quiet = False):
    if lambda_control is not None:
        raise ValueError("Input 'lambda_control' deprecated in v3.6.0; lambda is now selected by hyperparameter optimization")
    if not quiet:
        print(f"Input data has {self.dt_mod.shape[0]} {self.intervalType} in total: {self.dt_mod.ds.min()} to {self.dt_mod.ds.max()}")
        if "refreshDepth" in dir(self):
            depth = self.refreshDepth
        elif "refreshCounter" in dir(self):
            depth = self.refreshCounter
        else:
            depth = 0
        
        refresh = int(depth) > 0
        print("Model is built")
    if refresh:
        print("Rolling window moving forward")
    