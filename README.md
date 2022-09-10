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
    python3 -m venv .env && \
    source .env/bin/activate && \
    pip install -r base-linux.txt && \
    cd src && ./demo.py 
```