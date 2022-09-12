# Fully Homomorphic Encryption experiments


## Instruction to run the demo

* Download the kaggle CLI (https://github.com/Kaggle/kaggle-api)
* Execute the following:

```
git clone https://github.com/gabrielmougard/fhe-experiment.git

cd fhe-experiment && \
    mkdir -p datasets/medical-mnist && \
    kaggle datasets download -p datasets/medical-mnist andrewmvd/medical-mnist && \
    cd datasets/medical-mnist && unzip medical-mnist.zip && rm medical-mnist.zip && cd ../.. && \
```

Then to build and launch the demo:

```
cd fhe-experiment && python3 -m venv .env && \
    source .env/bin/activate && \
    pip install -r base-linux.txt && \
    cd src && ./demo.py 
```

The instruction above is made for training (GPU included) + inference. However,
I already provide a trained model (rtx 3060, 99% acc.) so if you just
want to launch the demo for inference you can use docker (as the zama's
Concrete compiler is somehow tricky to install with pip) :

```
cd fhe-experiment && docker build -t fhe-demo . && ./launch.sh
```
