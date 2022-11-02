from ..Dev import DevLoss
import torch
import torch.nn as nn


class CycleGANLoss(DevLoss):
    def __init__(
            self,
            image_size_tuple: tuple,
            n_layers_d: int,
            lambda_cycle: float,
            lambda_identity: float

    ):
        super().__init__()
        self.img_height, self.img_width = image_size_tuple
        self.n_layers_d = n_layers_d
        self.patch = (1,
                      self.img_height // (2 ** self.n_layers_d) - 2,
                      self.img_width // (2 ** self.n_layers_d) - 2)

        self.loss_func_gan = nn.MSELoss()
        self.loss_func_cycle = nn.L1Loss()
        self.loss_func_identity = nn.L1Loss()

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compute_generator_loss(
            self,
            model_output: dict,
            *args,
            **kwargs
    ) -> dict:

        identity_a, identity_b = model_output.get('identity_a'), model_output.get('identity_b')
        real_a, real_b = model_output.get('real_a'), model_output.get('real_b')
        fake_a_predict, fake_b_predict = model_output.get('fake_a_predict'), model_output.get('fake_b_predict')
        recover_a, recover_b = model_output.get('recover_a'), model_output.get('recover_b')

        # Identity loss
        loss_identity = (self.loss_func_identity(identity_a, real_a) + self.loss_func_identity(identity_b, real_b)) / 2

        # GAN loss
        targets_for_g = torch.ones(size=(real_a.shape[0], *self.patch)).to(real_a.device)

        loss_gan = (self.loss_func_gan(fake_a_predict, targets_for_g) + self.loss_func_gan(fake_b_predict, targets_for_g)) / 2

        # Cycle loss

        loss_cycle = (self.loss_func_cycle(recover_a, real_a) + self.loss_func_cycle(recover_b, real_b)) / 2

        # total loss
        loss = loss_gan + self.lambda_cycle * loss_cycle + self.lambda_identity * loss_identity  # type:torch.Tensor
        return {
            'total_loss': loss,
            'loss_identity': loss_identity,
            'loss_gan': loss_gan,
            'loss_cycle': loss_cycle
        }

    def compute_discriminator_loss(
            self,
            model_output: dict,
            *args,
            **kwargs
    ) -> dict:
        fake_a_predict, fake_b_predict = model_output.get('fake_a_predict'), model_output.get('fake_b_predict')
        real_a_predict, real_b_predict = model_output.get('real_a_predict'), model_output.get('real_b_predict')
        batch_size = fake_a_predict.shape[0]
        device = fake_a_predict.device
        # train d_a
        targets_for_d_real = torch.ones(size=(batch_size, *self.patch)).to(device)
        targets_for_d_fake = torch.zeros(size=(batch_size, *self.patch)).to(device)

        loss_d_a = 0.5 * (self.loss_func_gan(fake_a_predict,
                                             targets_for_d_fake) + self.loss_func_gan(real_a_predict,
                                                                                      targets_for_d_real))

        # train d_b

        loss_d_b = 0.5 * (self.loss_func_gan(fake_b_predict,
                                             targets_for_d_fake) + self.loss_func_gan(real_b_predict,
                                                                                      targets_for_d_real))
        return {
            # "total_loss": ,  # be careful here!!!!!
            "loss_d_a": loss_d_a,
            "loss_d_b": loss_d_b,
        }
