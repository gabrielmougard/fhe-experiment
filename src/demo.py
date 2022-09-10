#!/usr/bin/env python3

import gradio as gr
import os

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from fhe_train import get_compiled_circuit_with_test_data, LABELS_MAP


def main():
    fhe_model, examples_map = get_compiled_circuit_with_test_data()
    quantized_examples_data = np.load("ui_examples/quantized_data.npy", allow_pickle=True)

    def fhe_encrypt(clear_input):
        """
        Encrypt `clear_input` and returns a `gr.Image` with encrypted data.
        """
        #input = Image.fromarray(clear_input)
        transform=transforms.Compose([
            transforms.PILToTensor(),
        ])
        img_tensor = transform(clear_input)
        img_tensor = img_tensor[None, :]
        #encrypted = fhe_model.forward_fhe.encrypt(fhe_model.quantize_input(img_tensor.numpy()).astype(np.uint8))
        transform = transforms.ToPILImage()
        return transform(torch.rand(3,60,60))


    def fhe_inference(input_img_name, encrypted_img, target):
        """
        Make the prediction using the FHECircuit and the encrypted data.
        """
        quantized_img = quantized_examples_data.item().get(input_img_name)
        out_fhe = fhe_model.forward_fhe.encrypt_run_decrypt(quantized_img)
        output = fhe_model.dequantize_output(out_fhe)
        y_pred = np.argmax(output, 1)
        return f"## Prediction : {LABELS_MAP[y_pred.item()]} \nTruth : {target}"


    with gr.Blocks() as demo:    
        gr.Markdown("## Fully Homomorphic Encryption and Deep Learning")
        with gr.Row():
            with gr.Blocks():
                input_img = gr.Image(label="clear image", shape=(60,60), image_mode="L", type="pil")
                input_img_name = gr.Textbox(visible=False)
                input_img_target = gr.Textbox(label="ground truth")
            encrypted_img = gr.Image(label="encrypted image")
            prediction_res = gr.Markdown()
            
        with gr.Row():
            encrypt_btn = gr.Button(value="Encrypt")
            predict_btn = gr.Button(value="Predict")
        
        encrypt_btn.click(fhe_encrypt, inputs=input_img, outputs=encrypted_img)
        predict_btn.click(fhe_inference, inputs=[input_img_name, encrypted_img, input_img_target], outputs=prediction_res)
            

        gr.Markdown("## Test images (unencrypted)")
        gr.Examples(
            inputs=[
                input_img,
                input_img_name,
                input_img_target,
                encrypted_img,
            ],
            examples=[
                [os.path.join(os.path.dirname(__file__), f"ui_examples/{filename}"), filename, LABELS_MAP[target.item()], None] for filename, target in examples_map.items()
            ],
            outputs=[
                prediction_res
            ]
        )


    demo.launch()


if __name__ == "__main__":
    main()