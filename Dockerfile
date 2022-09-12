FROM zamafhe/concrete-numpy:latest


# Install dependencies (we have to do it manually to counter the firewall with the pip registry)
RUN pip install tqdm gradio onnx scipy torch torchvision torchsummary

# Run the demo
COPY src /usr/src/app/fhe-experiment/src
COPY checkpoints /usr/src/app/fhe-experiment/checkpoints
COPY datasets /usr/src/app/fhe-experiment/datasets

WORKDIR "/usr/src/app/fhe-experiment"

EXPOSE 7860

CMD ["/usr/bin/python", "-u", "src/demo.py"]
