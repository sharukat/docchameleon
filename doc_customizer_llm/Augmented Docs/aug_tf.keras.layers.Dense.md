description: Just your regular densely-connected NN layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Dense" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Dense

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/core/dense.py#L33-L301">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 65%;">

Just your regular densely-connected NN layer.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=&#x27;glorot_uniform&#x27;,
    bias_initializer=&#x27;zeros&#x27;,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

`Dense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`). These are all attributes of
`Dense`.

Note: If the input to the layer has a rank greater than 2, then `Dense`
computes the dot product between the `inputs` and the `kernel` along the
last axis of the `inputs` and axis 0 of the `kernel` (using <a href="../../../tf/tensordot.md"><code>tf.tensordot</code></a>).
For example, if input has dimensions `(batch_size, d0, d1)`, then we create
a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
`batch_size * d0` such sub-tensors).  The output in this case will have
shape `(batch_size, d0, units)`.

Besides, layer attributes cannot be modified after the layer has been called
once (except the `trainable` attribute).
When a popular kwarg `input_shape` is passed, then keras will create
an input layer to insert before the current layer. This can be treated
equivalent to explicitly defining an `InputLayer`.

#### Example:



```
>>> # Create a `Sequential` model and add a Dense layer as the first layer.
>>> model = tf.keras.models.Sequential()
>>> model.add(tf.keras.Input(shape=(16,)))
>>> model.add(tf.keras.layers.Dense(32, activation='relu'))
>>> # Now the model will take as input arrays of shape (None, 16)
>>> # and output arrays of shape (None, 32).
>>> # Note that after the first layer, you don't need to specify
>>> # the size of the input anymore:
>>> model.add(tf.keras.layers.Dense(32))
>>> model.output_shape
(None, 32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`units`<a id="units"></a>
</td>
<td>
Positive integer, dimensionality of the output space.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
Activation function to use.
If you don't specify anything, no activation is applied
(ie. "linear" activation: `a(x) = x`).
</td>
</tr><tr>
<td>
`use_bias`<a id="use_bias"></a>
</td>
<td>
Boolean, whether the layer uses a bias vector.
</td>
</tr><tr>
<td>
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>
Initializer for the `kernel` weights matrix.
</td>
</tr><tr>
<td>
`bias_initializer`<a id="bias_initializer"></a>
</td>
<td>
Initializer for the bias vector.
</td>
</tr><tr>
<td>
`kernel_regularizer`<a id="kernel_regularizer"></a>
</td>
<td>
Regularizer function applied to
the `kernel` weights matrix.
</td>
</tr><tr>
<td>
`bias_regularizer`<a id="bias_regularizer"></a>
</td>
<td>
Regularizer function applied to the bias vector.
</td>
</tr><tr>
<td>
`activity_regularizer`<a id="activity_regularizer"></a>
</td>
<td>
Regularizer function applied to
the output of the layer (its "activation").
</td>
</tr><tr>
<td>
`kernel_constraint`<a id="kernel_constraint"></a>
</td>
<td>
Constraint function applied to
the `kernel` weights matrix.
</td>
</tr><tr>
<td>
`bias_constraint`<a id="bias_constraint"></a>
</td>
<td>
Constraint function applied to the bias vector.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
N-D tensor with shape: `(batch_size, ..., input_dim)`.
The most common situation would be
a 2D input with shape `(batch_size, input_dim)`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
N-D tensor with shape: `(batch_size, ..., units)`.
For instance, for a 2D input with shape `(batch_size, input_dim)`,
the output would have shape `(batch_size, units)`.
</td>
</tr>

</table>

</div>


<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px;">
  
# Augmented Content

### Question: How does tf.keras.layers.Dense handle kernels for inputs with rank greater than 2?

When using the `tf.keras.layers.Dense` layer with inputs that have a rank greater than 2, the layer computes the dot product between the inputs and the kernel along the last axis of the inputs and axis 0 of the kernel. For instance, if the input has dimensions `(batch_size, d0, d1)`, a kernel with shape `(d1, units)` is created. This kernel operates along axis 2 of the input, on every sub-tensor of shape `(1, 1, d1)`, resulting in an output shape of `(batch_size, d0, units)`.

For inputs with a rank greater than 2, only one kernel is created, and this same kernel is applied to all slices of the second dimension. Consequently, the outputs for different indices of the second dimension are not independent of each other, especially during training.

To achieve independence across different indices of the second dimension, you would need to define a custom Keras layer. This custom layer would involve creating a stack of kernels instead of a single kernel and performing the specified tensor multiplication using `tf.einsum`.


```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class CustomDense(layers.Layer):
    def __init__(self, units):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        d0, i0 = input_shape[1], input_shape[2]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(d0, i0, self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(d0, self.units),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        # Perform the tensor multiplication using tf.einsum
        output = tf.einsum('abc,bcg->abg', inputs, self.kernel)
        return output + self.bias

# Example usage
input_data = np.random.random((10, 2, 3)).astype(np.float32)
inputs = tf.keras.Input(shape=(2, 3))
outputs = CustomDense(4)(inputs)
model = tf.keras.Model(inputs, outputs)

# Print the model summary
model.summary()

# Get the output of the model
output_data = model(input_data)
print(output_data)
```

### Related YouTube Tutorials
<a href="https://www.youtube.com/watch?v=oJ1i2c1KxKk" target="_blank">Python Tutorial: Keras input and dense layers</a>

<a href="https://www.youtube.com/watch?v=lor2LnEVn8M" target="_blank">[TensorFlow 2 Deep Learning] Dense Layer</a>

<a href="https://www.youtube.com/watch?v=kyktbJpg2mU" target="_blank">[Sequential Model, Dense Layer, and Model Compile in Keras Deep Learning</a>


### Related Stack Overflow Posts
<a href="https://stackoverflow.com/questions/68984841/how-can-i-understand-the-kernel-of-tf-keras-layers-dense-for-rank-2" target="_blank">How can I understand the kernel of tf.keras.layers.Dense for rank >2</a>

<a href="https://stackoverflow.com/questions/52089601/keras-dense-layers-input-is-not-flattened" target="_blank">Keras Dense layer's input is not flattened</a>

<a href="https://stackoverflow.com/questions/63507023/how-to-make-a-keras-dense-layer-deal-with-3d-tensor-as-input-for-this-softmax-fu" target="_blank">How to make a Keras Dense Layer deal with 3D tensor as input for this Softmax Fully Connected Layer?</a>

</div>
