
# !pip install torch torchvision matplotlib tensorboard -q
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading config from json file. 
def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("LOADED CONFIGURATION")
    #print(json.dumps(config, indent=2))
    return config

def save_config(config, save_path='run_config.json'):
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {save_path}")

# Load configuration
config = load_config('config.json')

exp_name = config.get('experiment_name', 'default')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f"{exp_name}_{timestamp}"

# Build paths
base_dir = "./runs"
run_dir = os.path.join(base_dir, run_name)

config['paths']['checkpoint_dir'] = os.path.join(run_dir, 'checkpoints')
config['paths']['tensorboard_dir'] = os.path.join(run_dir, 'tensorboard')

print(f"Run directory: {run_dir}")

# Create necessary directories
os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
os.makedirs(config['paths']['tensorboard_dir'], exist_ok=True)

# STEP 2
def build_transforms(config):
    aug_config = config['data_augmentation']
    norm_config = aug_config['normalize']
    
    # Training transforms
    train_transforms = []
    
    if aug_config['train']['random_horizontal_flip']:
        train_transforms.append(transforms.RandomHorizontalFlip())
    
    if 'random_crop' in aug_config['train']:
        crop_config = aug_config['train']['random_crop']
        train_transforms.append(
            transforms.RandomCrop(crop_config['size'], padding=crop_config['padding'])
        )
    
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_config['mean'], std=norm_config['std'])
    ])
    
    # Test transforms (no augmentation)
    test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_config['mean'], std=norm_config['std'])
    ]
    
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)

# Build transforms from config
transform_train, transform_test = build_transforms(config)

# Load dataset using paths from config
trainset = torchvision.datasets.ImageFolder(
    root=config['paths']['train_data'], 
    transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=config['training']['batch_size'], 
    shuffle=True, 
    num_workers=config['training']['num_workers']
)

testset = torchvision.datasets.ImageFolder(
    root=config['paths']['test_data'], 
    transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=config['training']['batch_size'], 
    shuffle=False, 
    num_workers=config['training']['num_workers']
)

print(f"Number of training classes: {len(trainset.classes)}")
print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, config['model']['num_classes'])
model = model.to(device)

def build_criterion(config):
    loss_type = config['loss']['type']
    
    if loss_type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss {loss_type} not supported")

def build_optimizer(model, config):
    opt_config = config['optimizer']
    opt_type = opt_config['type']
    
    if opt_type == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            momentum=opt_config['momentum'],
            weight_decay=opt_config['weight_decay']
        )
    elif opt_type == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Optimizer {opt_type} not supported")

def build_scheduler(optimizer, config):
    sched_config = config['scheduler']
    sched_type = sched_config['type']
    
    if sched_type == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    elif sched_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max']
        )
    else:
        raise ValueError(f"Scheduler {sched_type} not supported")

criterion = build_criterion(config)
optimizer = build_optimizer(model, config)
scheduler = build_scheduler(optimizer, config)

print(f"Loss: {config['loss']['type']}")
print(f"Optimizer: {config['optimizer']['type']}")
print(f"Scheduler: {config['scheduler']['type']}")


writer = SummaryWriter(config['paths']['tensorboard_dir'])
iteration = 0

# Save the config used for this run
save_config(config, os.path.join(config['paths']['checkpoint_dir'], 'run_config.json'))

num_epochs = config['training']['num_epochs']
train_losses, train_acc_list, test_acc_list = [], [], []

print("STARTING TRAINING")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Log to TensorBoard every iteration
        writer.add_scalar('Loss/train', loss.item(), iteration)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], iteration)
        
        # Save images every N iterations (from config)
        if iteration % config['logging']['save_images_every'] == 0:
            save_path = os.path.join(config['paths']['checkpoint_dir'], f'batch_{iteration}.png')
            torchvision.utils.save_image(inputs[:16], save_path, normalize=True)
            print(f'Saved {save_path}')
        
        iteration += 1
    
    train_loss = running_loss / len(trainloader.dataset)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total
    test_acc_list.append(test_acc)
    
    # Log epoch metrics to TensorBoard
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('Loss/train_epoch', train_loss, epoch)
    
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | '
          f'Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | '
          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Save checkpoint every N epochs (from config)
    if (epoch + 1) % config['logging']['save_checkpoint_every'] == 0:
        checkpoint_path = os.path.join(
            config['paths']['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_acc': test_acc,
            'config': config
        }, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

writer.close()

# Evaluate and save predictions on test set
print("\nGenerating predictions on test set...")
model.eval()
all_predictions = []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        for i in range(len(labels)):
            all_predictions.append({
                'true_class': trainset.classes[labels[i].item()],
                'predicted_class': trainset.classes[predicted[i].item()],
                'confidence': confidences[i].item() * 100,
                'correct': (predicted[i] == labels[i]).item()
            })

# Save predictions
predictions_path = os.path.join(config['paths']['checkpoint_dir'], 'test_predictions.json')
with open(predictions_path, 'w') as f:
    json.dump(all_predictions, f, indent=2)

print(f'Predictions saved to: {predictions_path}')
print(f'Total test images: {len(all_predictions)}')
print(f'Correct: {sum(p["correct"] for p in all_predictions)}')

# Save final model
final_model_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f'\nFinal model saved to: {final_model_path}')

# step 7? plot it all out. 
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy')
plt.legend()

plot_path = os.path.join(config['paths']['checkpoint_dir'], 'training_plots.png')
plt.savefig(plot_path)
print(f'Training plots saved to: {plot_path}')
plt.show()

print("TRAINING COMPLETE")
print(f"To view TensorBoard: tensorboard --logdir={config['paths']['tensorboard_dir']}")
print(f"All outputs saved to: {config['paths']['checkpoint_dir']}")