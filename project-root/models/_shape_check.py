import torch
from models.autoencoder import KoopmanAutoencoder

model = KoopmanAutoencoder(state_dim=2, input_dim=1, lift_dim=8, hidden_dims=(32, 32))

batch_size = 5
x = torch.randn(batch_size, 2)
u = torch.randn(batch_size, 1)

out = model.forward(x, u)
for key, val in out.items():
    print(key, val.shape)


from models.losses import reconstruction_loss, linear_dynamics_consistency_loss, prediction_loss

x_next = torch.randn(batch_size, 2)
z_next = model.encode(x_next)

recon = reconstruction_loss(x, out["x_hat"])
dyn = linear_dynamics_consistency_loss(z_next, out["z_next_pred"])
pred = prediction_loss(x_next, out["x_next_hat"])

print("reconstruction_loss:", recon.item())
print("linear_dynamics_consistency_loss:", dyn.item())
print("prediction_loss:", pred.item())

from models.losses import total_koopman_loss

losses = total_koopman_loss(x, x_next, u, model, lambda_physics=1.0, benchmark="van_der_pol")
for key, val in losses.items():
    print(f"{key}: {val.item()}")