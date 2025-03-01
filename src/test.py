'''test.py'''

import matplotlib.pyplot as plt
import model
import dataset
import train

def visualize_results(train_loss, val_loss):
  plt.figure(figsize=(14, 6))
  plt.subplot(1,2,1)
  plt.plot(train_loss)
  plt.xlabel("iteration")
  plt.ylabel("train loss")
  plt.title("training loss")
  plt.subplot(1,2,2)
  plt.plot(val_loss)
  plt.xlabel("iteration")
  plt.ylabel("val loss")
  plt.title("validation loss")
  plt.show()

def view_reconstruction(model, val_loader, noise_model):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  val_images_noisy, val_images, _ = next(iter(val_loader))
  val_images_noisy = val_images_noisy.to(device)

  with torch.no_grad():
    reconstructed_images = model(val_images_noisy)

  val_images_noisy = val_images_noisy.cpu().numpy()
  reconstructed_images = reconstructed_images.cpu().numpy()
  fig, axes = plt.subplots(2, 4, figsize=(12, 8))
  for i in range(4):
    axes[0, i].imshow(val_images[i].permute(1,2,0).squeeze(), cmap="gray")
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(np.transpose(reconstructed_images[i],(1,2,0)), cmap="gray")
    axes[1, i].set_title("Noisy Reconstruction")
    axes[1, i].axis("off")

  plt.show()

gaussian = GaussianNoise(0, 0.01)
model = DenoisingAutoencoder()
cifar100_train_gaussian = NoisyDataset('./data', train=True, download=True, transform=transform, noise_model=gaussian)
train_loader_100 = DataLoader(cifar100_train_gaussian, batch_size=1024, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
val_loader_noisy = DataLoader(cifar100_train_gaussian, batch_size=1024, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))
print('Training with Gaussian noise...')
model, gaussian_train_loss, gaussian_val_loss = train_dae(model, train_loader_100, val_loader_noisy, gaussian, num_epochs=75)
visualize_results(gaussian_train_loss, gaussian_val_loss)
view_reconstruction(model, val_loader_noisy, gaussian)

masking = MaskingNoise(0.2, mean, std)
model = DenoisingAutoencoder()
cifar100_train_masking = NoisyDataset('./data', train=True, download=True, transform=transform, noise_model=masking)
train_loader_100 = DataLoader(cifar100_train_masking, batch_size=1024, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
val_loader_noisy = DataLoader(cifar100_train_masking, batch_size=1024, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))
print('Training with Masking noise...')
model, masking_train_loss, masking_val_loss = train_dae(model, train_loader_100, val_loader_noisy, masking, num_epochs=75)
visualize_results(masking_train_loss, masking_val_loss)
view_reconstruction(model, val_loader_noisy, masking)

salt_pepper = SaltPepperNoise(0.2, mean, std)
model = DenoisingAutoencoder()
cifar100_train_sp = NoisyDataset('./data', train=True, download=True, transform=transform, noise_model=salt_pepper)
train_loader_100 = DataLoader(cifar100_train_sp, batch_size=1024, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
val_loader_noisy = DataLoader(cifar100_train_sp, batch_size=1024, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))
print('Training with Salt-Pepper noise...')
model, salt_pepper_train_loss, salt_pepper_val_loss = train_dae(model, train_loader_100, val_loader_noisy, salt_pepper, num_epochs=75)
visualize_results(salt_pepper_train_loss, salt_pepper_val_loss)
view_reconstruction(model, val_loader_noisy, salt_pepper)


