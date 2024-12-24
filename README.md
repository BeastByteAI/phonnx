

> [!WARNING]
> PHONNX was deprecated in favor of [FNNX](https://github.com/BeastByteAI/fnnx)


<h1 align="center">
  <br>
 <img src="https://github.com/BeastByteAI/.github/blob/main/phonnx_logo.png?raw=true" alt="AgentDingo" width="200" height = "200">
  <br>
  PHONNX
  <br>
</h1>

<h4 align="center">Python Hypercharged ONNX Runtime</h4>

<p align="center">
  <a href="https://github.com/OKUA1/BeastByteAI/phonnx">
    <img src="https://img.shields.io/github/v/release/BeastByteAI/phonnx.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://discord.gg/YDAbwuWK7V">
    <img src="https://dcbadge.vercel.app/api/server/YDAbwuWK7V?compact=true&style=flat">
  </a>
</p>
## What is PHONNX?

PHONNX (pronounced as 'phoenix') is a python wrapper arround [ONNX Runtime](https://onnxruntime.ai/) that provides several additional features, such as custom operators or automatic input preprocessing.

**Q:** Why PHONNX ? <br>
**A:** We developed PHONNX to serve as both a submodule and a stand-alone inference tool, mainly to meet the requirements of our projects such as [Falcon](https://github.com/BeastByteAI/falcon) and [Scikit-LLM](https://github.com/iryna-kondr/scikit-llm). However, we believe that PHONNX can be useful as a general purpose wrapper for ONNX Runtime as well.

**Q:** Are ONNX and PHONNX mutually compatible ? <br>
**A:** Every _phonnx model_ follows regular ONNX format specification, but adds additional metadata through various (mainly naming) conventions. This means that phonnx models can be used with any ONNX compatible runtime (given that the custom operators are also implemented in that runtime), but will only provide the additional features when used with PHONNX. On the other hand, PHONNX is not able to run arbitrary ONNX models, as it requires certain metadata to be present.

## Installation

PHONNX can be installed from PyPI using pip:

```bash
pip install phonnx
```

## Usage

PHONNX is designed to be easy to use. The following example shows how to load a model and run inference:

```python
from phonnx.runtime import Runtime

rt = Runtime('path/to/model.onnx')

y = rt.run(inputs, dynattrs, outputs_to_return = "final")
```

- `inputs` can be one of the following:
  - A dictionary of input names and values (i.e. `{"falcon_input_feature1_0": np.array([[1.]]), "falcon_input_feature2_0": np.array([[2.]])}`) which will be mapped to the corresponding inputs of the model;
  - A list of values (i.e. `[np.array([[1.]]), np.array([[2.]])]`) which will be mapped to the inputs in the order they appear in the model;
  - A single numpy array (i.e. `np.array[[1., 2.]]`), which will be split into the list of arrays across the last dimension and mapped to the inputs in the order they appear in the model.
- `dynattrs` (optional): a dictionary of dynamic attribute names and values (i.e. `{"falcon_dynattr_key": np.array(["value"])}`) which will be mapped to the corresponding dynamic attributes of the model;
- `outputs_to_return` (optional): a string indicating which outputs to return. Can be one of the following:
  - `final` (default): returns only the outputs of the final layer;
  - `all`: returns all outputs;

## Core PHONNX-specific concepts

### 1. Inputs vs Dynamic Attributes

In PHONNX we distinguish between two types of inputs: _inputs_ and _dynamic attributes_. Inputs are the actual features that are used as inputs to the model, while dynamic attributes are additional information that is passed to the model, but is not part of the dataset. A dynamic attribute is always a string tensor with a single element, while an input can be of any type and shape. The dynamic attributes are used to provide additional information to the model during inference, and consumed by custom operators. If the dynamic attribute is not provided, it will be automatically set to a tensor consisting of a single empty string.

The differentiation between inputs and dynamic attributes is conducted based on the naming convention. All input nodes must match one of the following regex patterns:

- `^([^-|_]+)[-|_](input)[-|_](.*?)[-|_]([0-9]+)$` for model inputs (i.e. `prefix_input_feature_0`);
- `^([^-|_]+)[-|_](dynattr)[-|_](.*?)$` for dynamic attributes (i.e. `prefix_dynattr_key`).

### 2. Input types and preprocessing

In PHONNX each individual feature is fed to the model as a separate input. Each input can be of a different input type (importantly the input type is not the same as the tensor type). The input type is used to determine how the input is preprocessed before being fed to the model. The type itself is inferred from the input name, which must have a suffix matching the number of the input type. The input types are defined in the `phonnx.constants.ColumnTypes`.

For example, `prefix_input_feature_0` is an input of type 0, which corresponds to a numeric feature of a tabular dataset.

We always assume that the first dimension of the input tensor is the batch dimension, and the input tensor should have at least two dimensions. If this is not the case, the input tensor will be automatically reshaped to have two dimensions, with the first dimension being the batch dimension.

### 3. Nodes and layers

All non-input nodes (intermediate nodes and outputs) should have the following prefix `<prefix>_pl_<id>/` (i.e. `falcon_pl_0/`), where the `<id>` indicates the layer to which the node belongs. We do not explicitly define what a layer is, but rather leave it up to the model developer to decide. The main purpose of the layers is to provide a mechanism for filtering the outputs. For example, if the outputs of intermediate layers are not needed, they can be removed by setting `outputs_to_return` to `final` during inference. In this case only the outputs with the `max(<id>)` will be returned.

### 4. Custom operators

At the moment no custom operators are implemented in PHONNX. They will be added in the next releases under the `ai.beastbyte.*` namespaces/opsets.

