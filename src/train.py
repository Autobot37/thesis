import torch
import torch.nn as nn
import os


def train_point_model(model, train_loader, val_loader, epochs=20, lr=1e-3,
                      device='cuda', save_dir='/root/checkpoints'):
    """Train DLinear or PatchTST with MSE loss + early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience, wait = 10, 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0
        for x, y, c in train_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            loss = criterion(model(x, c), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, c in val_loader:
                x, y, c = x.to(device), y.to(device), c.to(device)
                val_loss += criterion(model(x, c), y).item()
        val_loss /= len(val_loader)

        print(f"  [{model.__class__.__name__}] Epoch {epoch+1}/{epochs}"
              f"  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'best_{model.__class__.__name__}.pt'))
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return model


def train_diffusion(model, train_loader, val_loader,
                    epochs_ar=20, epochs_diff=50, lr=1e-3,
                    device='cuda', save_dir='/root/checkpoints'):
    """
    Two-phase training for DiffusionTS:

    Phase 1 — pretrain the AR initialisation layer  (fast, few epochs).
    Phase 2 — train the full diffusion model end-to-end (AR params included).
    """
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    # ================================================================
    # Phase 1:  Pretrain AR linear layer
    # ================================================================
    print(f"  [DiffusionTS] Phase 1: Pretraining AR layer ({epochs_ar} epochs)")
    # Optimize both AR linear layer and its city bias
    ar_opt = torch.optim.Adam(
        list(model.ar_linear.parameters()) + list(model.ar_city_bias.parameters()), 
        lr=lr
    )
    criterion = nn.MSELoss()

    for epoch in range(epochs_ar):
        model.train()
        ar_loss_sum = 0
        for x, y, c in train_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            # x: [B, L, 1] -> squeeze -> [B, L]
            z_ar = model.ar_linear(x.squeeze(-1)) + model.ar_city_bias(c)  # [B, H]
            loss = criterion(z_ar, y.squeeze(-1))
            ar_opt.zero_grad()
            loss.backward()
            ar_opt.step()
            ar_loss_sum += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    AR epoch {epoch+1}/{epochs_ar}"
                  f"  loss={ar_loss_sum / len(train_loader):.5f}")

    # ================================================================
    # Phase 2:  Train full diffusion model
    # ================================================================
    print(f"  [DiffusionTS] Phase 2: Training diffusion ({epochs_diff} epochs)")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_diff)
    best_val_loss = float('inf')
    patience, wait = 15, 0

    for epoch in range(epochs_diff):
        # --- train ---
        model.train()
        train_loss = 0
        for x, y, c in train_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            loss = model(x, y, c)           # model.forward() returns the loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # --- validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, c in val_loader:
                x, y, c = x.to(device), y.to(device), c.to(device)
                val_loss += model(x, y, c).item()
        val_loss /= len(val_loader)

        print(f"    Diff epoch {epoch+1}/{epochs_diff}"
              f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_DiffusionTS.pt'))
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    return model
