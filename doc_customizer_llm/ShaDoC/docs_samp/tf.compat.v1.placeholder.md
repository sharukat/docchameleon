description: Inserts a placeholder for a tensor that will be always fed.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.placeholder" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.placeholder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Inserts a placeholder for a tensor that will be always fed.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.placeholder(
    dtype, shape=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is not compatible with eager execution and <a href="../../../tf/function.md"><code>tf.function</code></a>. To migrate
to TF2, rewrite the code to be compatible with eager execution. Check the
[migration
guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
on replacing `Session.run` calls. In TF2, you can just pass tensors directly
into ops and layers. If you want to explicitly set up your inputs, also see
[Keras functional API](https://www.tensorflow.org/guide/keras/functional) on
how to use <a href="../../../tf/keras/Input.md"><code>tf.keras.Input</code></a> to replace <a href="../../../tf/compat/v1/placeholder.md"><code>tf.compat.v1.placeholder</code></a>.
<a href="../../../tf/function.md"><code>tf.function</code></a> arguments also do the job of <a href="../../../tf/compat/v1/placeholder.md"><code>tf.compat.v1.placeholder</code></a>.
For more details please read [Better
performance with tf.function](https://www.tensorflow.org/guide/function).

 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

**Important**: This tensor will produce an error if evaluated. Its value must
be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`.

#### For example:



```python
x = tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.compat.v1.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The type of elements in the tensor to be fed.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
The shape of the tensor to be fed (optional). If the shape is not
specified, you can feed a tensor of any shape.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` that may be used as a handle for feeding a value, but not
evaluated directly.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
if eager execution is enabled
</td>
</tr>
</table>


