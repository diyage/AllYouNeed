from ..Dev import DevTrainer


class MTCNNTrainer(DevTrainer):
    def __init__(
            self,
    ):
        super().__init__()

    def train_p_net_one_epoch(
            self
    ):
        pass

    def train_r_net_one_epoch(
            self
    ):
        pass

    def train_o_net_one_epoch(
            self
    ):
        pass

    def train_one_epoch(
            self,
            train_net_type: str,
            *args,
            **kwargs
    ):
        assert train_net_type in ['p', 'r', 'o']
        if train_net_type == 'p':
            self.train_p_net_one_epoch()
        elif train_net_type == 'r':
            self.train_r_net_one_epoch()
        else:
            self.train_o_net_one_epoch()
