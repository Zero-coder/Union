class BalancedDataLoaderIterator:
    """Round-robin iterator over task dataloaders."""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.length = min(len(loader) for loader in dataloaders) * len(dataloaders)

    def __len__(self):
        return self.length

    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        for _ in range(min(len(loader) for loader in self.dataloaders)):
            for task_id, iterator in enumerate(iterators):
                yield next(iterator), task_id
