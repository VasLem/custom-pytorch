from custom_pytorch.external.adamwr.cyclic_scheduler import CyclicLRWithRestarts as _CyclicLRWithRestarts

class CyclicLRWithRestarts(_CyclicLRWithRestarts):

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100, t_mult=2,
                 last_epoch=-1, verbose=False, policy='cosine', policy_fn=None,
                 min_lr=1e-07, eta_on_restart_cb=None, eta_on_iteration_cb=None, gamma=1.0,
                 triangular_step=0.5, groups_to_apply_to=None):

        super().__init__(optimizer, batch_size, epoch_size,
         restart_period=restart_period, t_mult=t_mult, last_epoch=last_epoch,
         verbose=verbose, policy=policy, policy_fn=policy_fn, min_lr=min_lr,
         eta_on_restart_cb=eta_on_restart_cb, eta_on_iteration_cb=eta_on_iteration_cb,
         gamma=gamma, triangular_step=triangular_step)
        self.groups_to_apply_to = groups_to_apply_to
        if groups_to_apply_to is not None:
            try:
                self.groups_to_apply_to[0]
            except TypeError:
                self.groups_to_apply_to = [self.groups_to_apply_to]

    def batch_step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self._on_iteration()
            self.iteration += 1
            self.total_iterations += 1
        except (IndexError):
            raise StopIteration("Epoch size and batch size used in the "
                                "training loop and while initializing "
                                "scheduler should be the same.")

        for cnt, (param_group, (lr, weight_decay)) in enumerate(zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur))):
            if self.groups_to_apply_to is not None:
                if cnt not in self.groups_to_apply_to:
                    continue
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay