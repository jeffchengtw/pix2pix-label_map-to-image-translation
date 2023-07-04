from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.epoch_losses = {}
        self.batch_num = 0

    def log_losses(self, losses, epoch):
        for name, loss in losses.items():
            self.writer.add_scalar(name, loss, epoch)

    def log_epoch_losses(self, epoch):
        avg_losses = {k: v / self.batch_num for k, v in self.epoch_losses.items()}
        self.log_losses(avg_losses, epoch)
        self.epoch_losses = {}  # 重置 epoch_losses 字典
        self.batch_num = 0

    def update_epoch_losses(self, losses):
        for name, loss in losses.items():
            if name not in self.epoch_losses:
                self.epoch_losses[name] = 0.0
            self.epoch_losses[name] += loss
        self.batch_num += 1
    
    def log_learning_rates(self, learning_rates, epoch):
        for name, learning_rate in learning_rates.items():
            self.writer.add_scalar(f'Learning Rate/{name}', learning_rate, epoch)

    def close(self):
        self.writer.close()
