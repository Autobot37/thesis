import re

with open('src/cvae.py', 'r') as f:
    content = f.read()

# Add patience to train_cvae args
content = content.replace("save_dir='/root/checkpoints'):", "save_dir='/root/checkpoints', patience=5, model_name='best_CVAE.pt'):")
content = content.replace("save_dir='/data/checkpoints'):", "save_dir='/data/checkpoints', patience=5, model_name='best_CVAE.pt'):")

# Update train_cvae logic
new_logic = '''    import os
    from tqdm import tqdm
    import sys
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_elbo = float('inf')
    epochs_no_improve = 0
    
    pbar = tqdm(range(epochs), file=sys.stdout, desc=f"Training {model_name}")
    for epoch in pbar:
        model.train()'''

content = re.sub(r'    import os.*?for epoch in range\(epochs\):\n        model\.train\(\)', new_logic, content, flags=re.DOTALL)

# Update print and early stopping
end_logic = '''        val_true_elbo = val_loss_total + val_kl_total
        
        pbar.set_postfix({'val_loss': f'{val_loss_total:.4f}', 'val_elbo': f'{val_true_elbo:.4f}', 'beta': f'{beta:.2f}'})
        
        if val_true_elbo < best_val_elbo:
            best_val_elbo = val_true_elbo
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\\nEarly stopping at epoch {epoch+1}")
                break
            
    return model'''

content = re.sub(r'        val_true_elbo = val_loss_total \+ val_kl_total.*?return model', end_logic, content, flags=re.DOTALL)

with open('src/cvae.py', 'w') as f:
    f.write(content)
