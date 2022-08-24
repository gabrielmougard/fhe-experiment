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
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concrete.common.compilation import CompilationConfiguration
from concreteml.torch.compile import compile_torch_model

cfg_simu = CompilationConfiguration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,  # This is for our tests only, never use that in prod
    treat_warnings_as_errors=True,
    use_insecure_key_cache=False,
)

cfg_prod = CompilationConfiguration(
    dump_artifacts_on_unexpected_failures=False,
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


def plot_vlib_quantized_result(model, train_loader, test_dataloader):
    accs = []
    accum_bits = []
    for n_bits in range(2, 9):
        # Compile and test the network with the Virtual Lib on the whole test set
        q_module_vl = compile_torch_model(
            model,
            torch.cat([img for img, _ in train_loader],0),
            n_bits=n_bits,
            use_virtual_lib=True,
            compilation_configuration=cfg_simu,
        )

        accum_bits.append(q_module_vl.forward_fhe.get_max_bit_width())
        accs.append(
            test_with_concrete_virtual_lib(
                q_module_vl,
                test_dataloader,
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