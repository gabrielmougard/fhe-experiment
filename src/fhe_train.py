# Copyright 2022 gab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchsummary import summary

from concrete.common.compilation import CompilationConfiguration
from concreteml.torch.compile import compile_torch_model

LABELS_MAP = {0: "Abdomen", 1: "Breast", 2: "Chest X-Ray", 3: "Chest computed tomography", 4: "Hand", 5: "Head"}

class TinyCNN(nn.Module):
    """A very small CNN to classify the medmsnist v1 dataset.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help keep the accumulator bit width low.

    This CNN only uses Conv2D, Dense layers and basic activation functions. Indeed,
    MaxPool2D/AvgPool2D are not yet supported operations for the Concrete compiler.
    We could implement this ONNX op as this is the first intermediary representation
    used by Concrete, but this demo is made to be simple...
    """

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2, 3, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(3, 16, 2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(16, 16, 2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(16, 3, 2, stride=2, padding=0)


        self.fc1 = nn.Linear(3*6*6, n_classes)

        # Enable pruning, prepared for training
        self.toggle_pruning(True)

    def toggle_pruning(self, enable):
        """Enables or removes pruning."""

        # Maximum number of active neurons (i.e. corresponding weight != 0)
        n_active = 10

        # Go through all the convolution layers
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            s = layer.weight.shape

            # Compute fan-in (number of inputs to a neuron)
            # and fan-out (number of neurons in the layer)
            st = [s[0], np.prod(s[1:])]

            # The number of input neurons (fan-in) is the product of
            # the kernel width x height x inChannels.
            if st[1] > n_active:
                if enable:
                    # This will create a forward hook to create a mask tensor that is multiplied
                    # with the weights during forward. The mask will contain 0s or 1s
                    prune.l1_unstructured(layer, "weight", (st[1] - n_active) * st[0])
                else:
                    # When disabling pruning, the mask is multiplied with the weights
                    # and the result is stored in the weights member
                    prune.remove(layer, "weight")

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = x.view(-1, 3*6*6)
        x = self.fc1(x)
        return x


def train_one_epoch(net, optimizer, train_loader, device):
    """Training loop for one epoch"""
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        output = net(data.to(device))
        loss_net = loss(output, target.long().to(device))
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    return avg_loss / len(train_loader)


def test_torch(net, test_loader, device):
    """Testing loop for one epoch"""
    net.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += loss_func(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_with_concrete_virtual_lib(quantized_module, test_loader, use_fhe, use_vl):
    """Testing loop for a model that is quantized and compiled with Concrete"""

    # When running in FHE, we cast inputs to uint8, but when running using the Virtual Lib (VL)
    # we may want inputs to exceed 8b to test quantization performance. Thus,
    # for VL we cast to int32
    dtype_inputs = np.uint8 if use_fhe else np.int32

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    correct = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        # Quantize the inputs and cast to appropriate data type
        x_test_q = quantized_module.quantize_input(data).astype(dtype_inputs)

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
            print(f"y_pred: {y_pred[0]} == {target[i].numpy()}")
            if y_pred[0] == target[i].numpy():
                correct += 1

    if use_vl and not use_fhe:
        test_conf = "Virtual FHE"
    elif use_fhe and not use_vl:
        test_conf = "FHE"
    elif not use_fhe and not use_vl:
        test_conf = "Quantized"
    else:
        test_conf = "FHE (override)"

    print('\nTest set (with {}): Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_conf, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )
        

def generate_demo_data(test_loader, q_module_vl):
    """
    Generate example data for the gradio UI (the first batch of the test set)
    """
    for file in os.listdir("ui_examples"):
        os.remove(f"ui_examples/{file}")

    saved_example = {}
    saved_quantized_data = {}
    for data, target in test_loader:
        data = data.numpy()
        x_test_q = q_module_vl.quantize_input(data).astype(np.int32)

        # Iterate over single inputs
        for i in range(data.shape[0]):
            filename = f"example_{LABELS_MAP[target[i].item()]}_{i}.png"                
            save_image(torch.from_numpy(data[i]), f"ui_examples/{filename}")
            x_q = np.expand_dims(x_test_q[i, :], 0)
            saved_quantized_data[filename] = x_q
            saved_example[filename] = target[i]
        break

    np.save("ui_examples/quantized_data.npy", saved_quantized_data)
    return saved_example


def get_compiled_circuit_with_test_data():

    BATCH_SIZE = 64
    N_EPOCHS = 30
    data_dir = '../datasets/medical-mnist'
    models_dir = '../checkpoints'

    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Convert image to grayscale (allow less complex model and faster FHE compilation)
        transforms.RandomRotation(10),               # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),           # reverse 50% of images
        transforms.Resize(60),                       # resize shortest side to 100 pixels
        transforms.CenterCrop(60),                   # crop longest side to 100 pixels at center
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5,],
            [0.5,]
        )
    ])

    dataset = ImageFolder(data_dir, transform=transform)
    print('Size of training dataset :', len(dataset))

    val_size = len(dataset)//10
    test_size = len(dataset)//5
    train_size = len(dataset) - val_size -test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    print(f"train dataset length: {len(train_ds)}, validation dataset length: {len(val_ds)}, test dataset length: {len(test_ds)}")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the tiny CNN with 7 output classes
    model = TinyCNN(7).to(device)
    # Show model summary
    summary(model, (1,60,60))


    # Classic torch training
    if not os.path.exists(f"{models_dir}/medmnist_cnn.pt"):
        # Train the network with Adam, output the test set accuracy every epoch
        optimizer = torch.optim.Adam(model.parameters())
        losses = []
        for e in range(N_EPOCHS):
            start_time = time.time()
            losses.append(train_one_epoch(model, optimizer, train_loader, device))
            print(f"epoch: {e} in {time.time() - start_time} seconds")
            test_torch(model, test_loader, device)

        # Finally, disable pruning (sets the pruned weights to 0)
        torch.save(model.state_dict(), f"{models_dir}/medmnist_cnn.pt")
        model.toggle_pruning(False)

        fig = plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.ylabel("Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.title("Training set loss during training")
        plt.show()
    else:
        model.load_state_dict(torch.load(f"{models_dir}/medmnist_cnn.pt", map_location=device))
        model.toggle_pruning(False)
        print("Model detected on disk, model loaded.")


    fhe_cfg_simu = CompilationConfiguration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
        treat_warnings_as_errors=True,
        use_insecure_key_cache=False,
    )

    # fhe_cfg_prod = CompilationConfiguration(
    #     dump_artifacts_on_unexpected_failures=False,
    #     treat_warnings_as_errors=True,
    #     use_insecure_key_cache=False,
    # )

    start_time = time.time() 
    print(f"Compiling to FHE representation...")
    for n_bit in range(8, 1, -1):
        try:
            print(f"Attempting to compile with {n_bit} precision...")
            q_module_vl = compile_torch_model(
                model.to("cpu"),
                torch.cat([img.to("cpu") for img, _ in train_loader],0)[0:200], # concatenate a part of the training set into one tensor as Tensor tuple is not yet supported.
                n_bits=n_bit,
                use_virtual_lib=True,
                compilation_configuration=fhe_cfg_simu,
            )
            print(f"Compiled FHE friendly torch model for {n_bit} quantization bits in {time.time() - start_time} seconds") # ~20min : AMD Ryzen Threadripper 2920X - 12 cores, 24 threads + 32 Gb RAM 
            break
        except RuntimeError as e:
            if str(e).startswith("max_bit_width of some nodes is too high"):
                print("The network is not fully FHE friendly, retraining.")
                continue
            raise e


    path = q_module_vl.forward_fhe.draw()
    print(f"FHECircuit graph generated at {path}")
    
    # FHE module serialization is not supported yet.
    # Although I tried to make it work, I'm facing serialization issues
    # with internal tricky JIT compiler logic. Better wait for an official support.
    # test_with_concrete_virtual_lib(
    #     q_module_vl,
    #     test_loader,
    #     use_fhe=False,
    #     use_vl=True,
    # ) # We achieve 98% accuracy on the test dataset which is only a drop of 1.2% from the original model.

    saved_examples = generate_demo_data(test_loader, q_module_vl)

    return q_module_vl, saved_examples
