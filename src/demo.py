import gradio as gr
import os

def combine(a, b):
    return a + " " + b

def encode(x):
    return x

with gr.Blocks() as demo:    
    with gr.Row():
        im = gr.Image()
        im_2 = gr.Image()
        
    btn = gr.Button(value="Encode")
    btn.click(encode, inputs=[im], outputs=[im_2])
        
    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "lion.jpeg")],
        inputs=im,
        outputs=im_2,
        fn=encode,
        cache_examples=True)

if __name__ == "__main__":
    demo.launch()