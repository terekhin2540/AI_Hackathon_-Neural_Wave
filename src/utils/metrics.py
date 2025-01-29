def accuracy(preds, y):
    return (preds == y).float().mean()
