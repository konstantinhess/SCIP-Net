import torch
from torch import nn
import torchcde


class CDEIntegrand(nn.Module):
    """
    Neural CDE integrand
    """
    def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0, inhomogeneous=True):
        super().__init__()
        # Input layer
        self.cde_layers = [nn.Linear(hidden_size+inhomogeneous, hidden_size),
                           nn.ReLU(),
                           nn.Dropout(dropout_rate)]
        # Hidden layers
        for _ in range(num_layer):
            self.cde_layers += [nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(), nn.Dropout(dropout_rate)
                                ]
        # Output layer
        self.cde_layers += [nn.Linear(hidden_size, hidden_size * input_size),
                            nn.Tanh(),
                            nn.Dropout(dropout_rate)]
        self.cde_layers = nn.Sequential(*self.cde_layers)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.inhomogeneous = inhomogeneous

    def forward(self, t, x):
        if self.inhomogeneous:
            x = torch.cat((x, t.unsqueeze(0).expand(x.size(0), 1)), dim=-1)
        return self.cde_layers(x).view(-1, self.hidden_size, self.input_size)


class NeuralCDE(nn.Module):
    """
    Neural CDE model
    """
    def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0, inhomogeneous=True):
        super().__init__()

        # Neural CDE integrand
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.f = CDEIntegrand(input_size, hidden_size, num_layer, dropout_rate, inhomogeneous=inhomogeneous)

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def forward(self, x, init_states=None, t_max=None, device='cpu'):

        if x.size(1) == 1:
            # Riemann-Stiltjes integral over singleton
            x[torch.isnan(x)] = 0 # will be masked away in training anyway; only to avoid weird bug in optimization
            return torch.matmul(self.f(torch.tensor(0, device=device), init_states), x.squeeze(1).unsqueeze(-1)).unsqueeze(1).squeeze(-1)
        else:

            if init_states is None:
                init_states = self.input_layer(x[:, 0, :])
            # (Decoder receives init_states from encoder)

            t_grid = torch.linspace(0, 1, x.size(1), device=device)
            t_out  = t_grid if t_max is None else torch.tensor([0, t_max], device=device)
            if (t_out[-1]-t_out[0]) == 0:
                return torch.zeros(x.size(0), x.size(1), init_states.size(1))
            else:
                coeffs = torchcde.linear_interpolation_coeffs(x, t_grid)
                X = torchcde.LinearInterpolation(coeffs, t_grid)
                # coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x, t_grid)
                # X = torchcde.CubicSpline(coeffs, t_grid)

                return torchcde.cdeint(X=X, func=self.f, z0=init_states, method='euler', t=t_out, #t_grid,
                                      adjoint=False)

