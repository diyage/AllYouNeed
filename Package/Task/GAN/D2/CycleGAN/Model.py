"""
be careful, Cycle-GAN is different from original GAN and W-GAN(do not inherit from xxModel).
There are four nets, two generators and two discriminators.
Situation:
    two generators:
        g_b_to_a --> feed real_b, return fake_a
                 --> feed fake_b, return recover_a
        g_a_to_b --> feed real_a, return fake_b
                 --> feed fake_a, return recover_b

    two discriminators:
        d_a     --> feed real_a, return real_a_predict
                --> feed fake_a, return fake_a_predict
        d_b     --> feed real_b, return real_b_predict
                --> feed fake_b, return fake_b_predict
"""

import torch
import torch.nn as nn


class CycleGANModel(nn.Module):
    def __init__(
            self,
            g_b_to_a: nn.Module,
            g_a_to_b: nn.Module,
            d_a: nn.Module,
            d_b: nn.Module
    ):
        super().__init__()
        self.g_b_to_a = g_b_to_a
        self.g_a_to_b = g_a_to_b
        self.d_a = d_a
        self.d_b = d_b
        """
        ----------------------------------------------------------------
        real_a --> g_b_to_a --> identity_a
        real_a --> g_a_to_b --> fake_b --> g_b_to_a --> recover_a
                                       --> d_b --> fake_b_predict
                                       
        real_a --> d_a --> real_a_predict
        ----------------------------------------------------------------
        real_b --> g_a_to_b --> identity_b
        real_b --> g_b_to_a --> fake_a --> g_a_to_b --> recover_b
                                       --> d_a --> fake_a_predict
        real_b --> d_b --> real_b_predict
        ----------------------------------------------------------------
        """

    def get_compute_generator_loss_need_info(
            self,
            real_a: torch.Tensor,
            real_b: torch.Tensor
    ) -> dict:
        identity_a = self.g_b_to_a(real_a)
        identity_b = self.g_a_to_b(real_b)

        fake_a = self.g_b_to_a(real_b)
        fake_a_predict = self.d_a(fake_a)

        fake_b = self.g_a_to_b(real_a)
        fake_b_predict = self.d_b(fake_b)

        recover_a = self.g_b_to_a(fake_b)
        recover_b = self.g_a_to_b(fake_a)
        return {
            'real_a': real_a,
            'real_b': real_b,
            'identity_a': identity_a,
            'identity_b': identity_b,
            'fake_a_predict': fake_a_predict,
            'fake_b_predict': fake_b_predict,
            'recover_a': recover_a,
            'recover_b': recover_b
        }

    def get_compute_discriminator_loss_need_info(
            self,
            real_a: torch.Tensor,
            real_b: torch.Tensor
    ) -> dict:
        fake_a = self.g_b_to_a(real_b)
        fake_a_predict = self.d_a(fake_a)
        real_a_predict = self.d_a(real_a)

        fake_b = self.g_a_to_b(real_a)
        fake_b_predict = self.d_b(fake_b)
        real_b_predict = self.d_b(real_b)
        return {
            'fake_a_predict': fake_a_predict,
            'real_a_predict': real_a_predict,
            'fake_b_predict': fake_b_predict,
            'real_b_predict': real_b_predict
        }

    def forward(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            'Please do not use __call__ of xxModel.'
        )
