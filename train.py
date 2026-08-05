import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import StampsDataset
from engine import train, test
from unet.u_net_model import U_net 
from utils.functions import plot_training_history

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_DIR = "../unet/"
CLEAN_DIR = "../data/stamps"   
NOISY_DIR = "../data/gaussian_noise" 

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # load dataset and split into training and testing 
    dataset = StampsDataset(clean_dir=CLEAN_DIR, noisy_dir=NOISY_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders (DATASET)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load model 
    model = U_net().to(device)
    loss_fn = nn.MSELoss() # future testing with MAE instead of MSE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_history = []
    val_loss_history = []

    # Run and test model
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Run training and test
        t_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        v_loss = test(val_dataloader, model, loss_fn, device)
    
        train_loss_history.append(t_loss)
        val_loss_history.append(v_loss)

    # save model learned
    save_path = f"{MODEL_DIR}/unet_denoising_pesos.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved PyTorch Model State to {save_path}")
    
    plot_training_history(train_loss_history, val_loss_history)

if __name__ == "__main__":
    main()