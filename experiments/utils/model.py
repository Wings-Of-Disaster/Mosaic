import torch

def evaluation(model, dataloader, criterion, model_params=None, device=None, eval_full_data=True):
    if model_params is not None:
        model.load_state_dict(model_params)

    if device is not None:
        model.to(device)

    model.eval()
    loss = 0.0
    acc = 0.0
    num = 0
    
    i = 0
    for x, y in dataloader:
        torch.cuda.empty_cache()
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        with torch.no_grad():
            logit = model(x) 
            _loss = criterion(logit, y)
            _, predicted = torch.max(logit, -1)
            _acc = predicted.eq(y).sum()
            _num = y.size(0)
            loss += (_loss * _num).item()
            acc += _acc.item()
            num += _num
            i += 1
            if not eval_full_data:
                if i == 10:
                    break
    loss /= num
    acc /= num
    return loss, acc, num


class CycleDataloader:
    def __init__(self, dataloader, epoch=-1, seed=None) -> None:
        self.dataloader = dataloader
        self.epoch = epoch
        self.seed = seed
        self._data_iter = None
        self._init_data_iter()

    def _init_data_iter(self):
        if self.epoch == 0:
            raise StopIteration()

        if self.seed is not None:
            torch.manual_seed(self.seed + self.epoch)
        self._data_iter = iter(self.dataloader)
        self.epoch -= 1

    def __next__(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._init_data_iter()
            return next(self._data_iter)

    def __iter__(self):
        return self