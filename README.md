<!--
 Copyright 2022 gab
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Fully Homomorphic Encryption experiments

## TODO for GC meetup:

* Benchmark medical MNIST v1 classification with `concrete`'s Quantized Module for different accumulators bit width
* Come up with an prod model (optimal bit accumulator width) to compile the `FHECircuit`
* Build the `gradio` app:
    * UI
    * context `concrete` key generation
    * FHE inference pipeline with `encrypt` ==> inference ==> `decrypt`

* Build a nice powerpoint :) 