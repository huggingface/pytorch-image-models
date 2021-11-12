class AvgMinMaxScalar:

    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.min = None
        self.max = None
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.min = val if self.min is None else min(self.min, val)
        self.max = val if self.max is None else max(self.max, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



