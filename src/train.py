import torch
import torch.nn as nn

def train_dae(model, train_loader, val_loader, noise, num_epochs=10):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # enable cuDNN benchmarking for performance
  torch.backends.cudnn.benchmark = True

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # add mixed precision training for better GPU utilization
  scaler = torch.cuda.amp.GradScaler()
  train_loss = []
  val_loss = []
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, clean_images, _ in train_loader:
      images=images.to(device, non_blocking = True)
      clean_images=clean_images.to(device, non_blocking = True)

      # use mixed precision training
      with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, clean_images)

      # backward pass
      optimizer.zero_grad(set_to_none=True) 
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      running_loss += loss.item()
    
    torch.cuda.empty_cache()
    tl = running_loss/len(train_loader)
    train_loss.append(tl)

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
      for images, clean_images, _ in val_loader:
        images = images.to(device, non_blocking = True)
        clean_images = clean_images.to(device, non_blocking = True)
        with torch.cuda.amp.autocast():
          reconstructed = model(images)
          loss = criterion(reconstructed, clean_images)
        running_loss += loss.item()

      vl = running_loss/len(val_loader)
      val_loss.append(vl)
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {tl:.4f}, Validation Loss: {vl:.4f}")
  
  return model, train_loss, val_loss

