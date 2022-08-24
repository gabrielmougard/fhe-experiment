from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils import prune
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from concrete.common.compilation import CompilationConfiguration
from concreteml.torch.compile import compile_torch_model

import medmnist
from medmnist import INFO, Evaluator


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        targets = labels.to(torch.float32)
        loss = nn.BCEWithLogitsLoss(out, targets) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        targets = labels.to(torch.float32)
        print(f"OUT : {out}")
        loss = nn.BCEWithLogitsLoss(out, targets)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class Model(ImageClassificationBase):
    def __init__(self, input_size, output_size):
        super().__init__()
        # hidden layer
        self.in_layer = nn.Linear(input_size, 8384)
        self.hidden1 = nn.Linear(8384, 4192)
        self.hidden2 = nn.Linear(4192, 2096)
        self.hidden3 = nn.Linear(2096, 1048)
        self.out_layer = nn.Linear(1048, output_size)

        # Enable pruning, prepared for training
        # self.toggle_pruning(True)

    # def toggle_pruning(self, enable):
    #     """Enables or removes pruning."""

    #     # Maximum number of active neurons (i.e. corresponding weight != 0)
    #     n_active = 10

    #     # Go through all the convolution layers
    #     for layer in [self.hidden1, self.hidden2, self.hidden3]:
    #         s = layer.weight.shape
    #         # Compute fan-in (number of inputs to a neuron)
    #         # and fan-out (number of neurons in the layer)
    #         st = [s[0], np.prod(s[1:])]
    #         # The number of input neurons (fan-in) is the product of
    #         # the kernel width x height x inChannels.
    #         if st[1] > n_active:
    #             if enable:
    #                 # This will create a forward hook to create a mask tensor that is multiplied
    #                 # with the weights during forward. The mask will contain 0s or 1s
    #                 prune.l1_unstructured(layer, "weight", (st[1] - n_active) * st[0])
    #             else:
    #                 # When disabling pruning, the mask is multiplied with the weights
    #                 # and the result is stored in the weights member
    #                 prune.remove(layer, "weight")
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        out = self.in_layer(out)
        out = self.hidden1(F.relu(out))
        out = self.hidden2(F.relu(out))
        out = self.hidden3(F.relu(out))
        out = self.out_layer(F.relu(out))
        return out

    
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()
    
    
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


data_flag = 'chestmnist'
download = True

NUM_EPOCHS = 1 # around 1min30 per epoch 
BATCH_SIZE = 64
lr = 0.001
input_size = 1*28*28
output_size = 14

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.RandomRotation(10),     # rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(), # reverse 50% of images
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

print(train_dataset)
print("===================")
print(test_dataset)

device = get_default_device()

# encapsulate data into dataloader with optimal device
train_loader = DeviceDataLoader(
    data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    device
)
train_loader_at_eval = DeviceDataLoader(
    data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False),
    device
)
test_loader = DeviceDataLoader(
    data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False),
    device
)

# visualization
# from torchvision.utils import save_image

# for i in range(len(test_dataset)):
#     if np.array_equal(test_dataset[i][1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]):
#         print("pnewumonia")
#         print(test_dataset[i])
#         img1 = test_dataset[i][0] #torch.Size([3,28,28]
#         # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
#         save_image(img1, 'img1.png')
#         break

# # define a simple CNN model
# class Net(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Net, self).__init__()

#         self.conv2Dlayers = [
#             nn.Conv2d(in_channels, 16, kernel_size=3),
#             nn.Conv2d(16, 16, kernel_size=3),
#             nn.Conv2d(16, 64, kernel_size=3),
#             nn.Conv2d(64, 64, kernel_size=3),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         ]

#         self.layer1 = nn.Sequential(
#             self.conv2Dlayers[0],
#             nn.BatchNorm2d(16),
#             nn.ReLU())

#         self.layer2 = nn.Sequential(
#             self.conv2Dlayers[1],
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.layer3 = nn.Sequential(
#             self.conv2Dlayers[2],
#             nn.BatchNorm2d(64),
#             nn.ReLU())
        
#         self.layer4 = nn.Sequential(
#             self.conv2Dlayers[3],
#             nn.BatchNorm2d(64),
#             nn.ReLU())

#         self.layer5 = nn.Sequential(
#             self.conv2Dlayers[4],
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.fc = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes))

#         # Enable pruning, prepared for training
#         self.toggle_pruning(True)

#     def toggle_pruning(self, enable):
#         """Enables or removes pruning."""

#         # Maximum number of active neurons (i.e. corresponding weight != 0)
#         n_active = 10

#         # Go through all the convolution layers
#         for layer in self.conv2Dlayers:
#             s = layer.weight.shape

#             # Compute fan-in (number of inputs to a neuron)
#             # and fan-out (number of neurons in the layer)
#             st = [s[0], np.prod(s[1:])]

#             # The number of input neurons (fan-in) is the product of
#             # the kernel width x height x inChannels.
#             if st[1] > n_active:
#                 if enable:
#                     # This will create a forward hook to create a mask tensor that is multiplied
#                     # with the weights during forward. The mask will contain 0s or 1s
#                     prune.l1_unstructured(layer, "weight", (st[1] - n_active) * st[0])
#                 else:
#                     # When disabling pruning, the mask is multiplied with the weights
#                     # and the result is stored in the weights member
#                     prune.remove(layer, "weight")

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# model = Net(in_channels=n_channels, num_classes=n_classes)

model = to_device(Model(input_size, output_size), device)


history = [evaluate(model, train_loader_at_eval)]
history += fit(NUM_EPOCHS, lr, model, train_loader, train_loader_at_eval)

plot_accuracies(history)
plot_losses(history)

# Finally, disable pruning (sets the pruned weights to 0)
# model.toggle_pruning(False)

##################### FHE circuit compilation ###################

cfg = CompilationConfiguration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,  # This is for our tests only, never use that in prod
    treat_warnings_as_errors=True,
    use_insecure_key_cache=False,
)

def test_with_concrete_virtual_lib(quantized_module, test_loader, use_fhe, use_vl):
    """Test a neural network that is quantized and compiled with Concrete-ML."""

    # When running in FHE, we cast inputs to uint8, but when running using the Virtual Lib (VL)
    # we may want inputs to exceed 8b to test quantization performance. Thus,
    # for VL we cast to int32
    dtype_inputs = np.uint8 if use_fhe else np.int32
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int32)
    all_targets = np.zeros((len(test_loader)), dtype=np.int32)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        # Quantize the inputs and cast to appropriate data type
        x_test_q = quantized_module.quantize_input(data).astype(dtype_inputs)

        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Iterate over single inputs
        for i in range(x_test_q.shape[0]):
            # Inputs must have size (N, C, H, W), we add the batch dimension with N=1
            x_q = np.expand_dims(x_test_q[i, :], 0)

            # Execute either in FHE (compiled or VL) or just in quantized
            if use_fhe or use_vl:
                out_fhe = quantized_module.forward_fhe.encrypt_run_decrypt(x_q)
                output = quantized_module.dequantize_output(out_fhe)
            else:
                output = quantized_module.forward_and_dequant(x_q)

            # Take the predicted class from the outputs and store it
            y_pred = np.argmax(output, 1)
            all_y_pred[idx] = y_pred
            idx += 1

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)
    return n_correct / len(test_loader)

accs = []
accum_bits = []
for n_bits in range(2, 9):
    # Compile and test the network with the Virtual Lib on the whole test set
    q_module_vl = compile_torch_model(
        model,
        tuple(t[0] for t in train_dataset),
        n_bits=n_bits,
        use_virtual_lib=True,
        compilation_configuration=cfg,
    )
    # pylint: disable=no-member
    accum_bits.append(q_module_vl.forward_fhe.get_max_bit_width())
    # pylint: enable=no-member

    accs.append(
        test_with_concrete_virtual_lib(
            q_module_vl,
            test_dataset,
            use_fhe=False,
            use_vl=True,
        )
    )

fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 14
plt.plot(range(2, 9), accs, "-x")
for bits, acc, accum in zip(range(2, 9), accs, accum_bits):
    plt.gca().annotate(str(accum), (bits - 0.1, acc + 0.025))
plt.ylabel("Accuracy on test set")
plt.xlabel("Weight & activation quantization")
plt.grid(True)
plt.title("Accuracy for varying quantization bit width")
plt.show()


#### real compilation 

q_module = compile_torch_model(
    model,
    train_dataset,
    n_bits=8,
    compilation_configuration=cfg,
)

### show FHE circuit
print(q_module.forward_fhe)