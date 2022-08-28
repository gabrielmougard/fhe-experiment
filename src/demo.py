import gradio as gr
import os

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from fhe_train import get_compiled_circuit_with_test_data


LABELS_MAP = {0: "Abdomen", 1: "Breast", 2: "Chest X-Ray", 3: "Chest computed tomography", 4: "Hand", 5: "Head"}

def generate_example_imgs(test_loader):
    """
    Save 18 example images on disk that should cover the 6 classes of the test dataset.
    Returns a mapping between the filename and the target label.
    """
    examples = {}
    example_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for batch, batch_target in test_loader:
        for im, target in zip(batch, batch_target):
            if example_counts[target.item()] < 3:
                filename = f"example_{LABELS_MAP[target.item()]}_{example_counts[target.item()]}.png"
                save_image(im, f"ui_examples/{filename}")
                examples[filename] = target
                example_counts[target.item()] += 1
            if sum([count for _, count in example_counts.items()]) == 18:
                return examples


def main():
    fhe_model, test_loader = get_compiled_circuit_with_test_data()
    
    torch.manual_seed(12538040293833290220)
    examples_map = generate_example_imgs(test_loader)

    def fhe_encrypt(clear_input):
        """
        Encrypt `clear_input` and returns a `gr.Image` with encrypted data.
        """
        input = Image.fromarray(clear_input)
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        img_tensor = transform(input)
        img_tensor = img_tensor[None, :]
        #encrypted = fhe_model.forward_fhe.encrypt(fhe_model.quantize_input(img_tensor.numpy()).astype(np.uint8))
        transform = transforms.ToPILImage()
        return transform(torch.rand(3,60,60))


    def fhe_inference(input_img, encrypted_img, target):
        """
        Make the prediction using the FHECircuit and the encrypted data.
    
        params:
            `input` (gr.Image) : the encrypted image 
        """
        input = Image.fromarray(input_img)
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        img_tensor = transform(input).numpy()
        x_q = fhe_model.quantize_input(img_tensor).astype(np.uint8)
        x_q = np.expand_dims(x_q[:], 0)
        out_fhe = fhe_model.forward_fhe.encrypt_run_decrypt(x_q)
        output = fhe_model.dequantize_output(out_fhe)
        y_pred = np.argmax(output, 1)
        
        return f"## Prediction : {LABELS_MAP[y_pred.item()]} \nTruth : {target}"


    with gr.Blocks() as demo:    

        gr.Markdown("## Fully Homomorphic Encryption and Deep Learning")
        with gr.Row():
            with gr.Blocks():
                input_img = gr.Image(label="clear image")
                input_img_target = gr.Textbox(label="ground truth")
            encrypted_img = gr.Image(label="encrypted image")
            prediction_res = gr.Markdown()
            
        with gr.Row():
            encrypt_btn = gr.Button(value="Encrypt")
            predict_btn = gr.Button(value="Predict")
        
        encrypt_btn.click(fhe_encrypt, inputs=input_img, outputs=encrypted_img)
        predict_btn.click(fhe_inference, inputs=[input_img, encrypted_img, input_img_target], outputs=prediction_res)
            

        gr.Markdown("## Test images (unencrypted)")
        gr.Examples(
            inputs=[
                input_img,
                input_img_target,
                encrypted_img,
            ],
            examples=[
                [os.path.join(os.path.dirname(__file__), f"ui_examples/{filename}"), LABELS_MAP[target.item()], None] for filename, target in examples_map.items()
            ],
            outputs=[
                prediction_res
            ]
        )


    demo.launch()


if __name__ == "__main__":
    main()