description: Computes and returns the sampled softmax training loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.sampled_softmax_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.sampled_softmax_loss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_impl.py">View source</a>



Computes and returns the sampled softmax training loss.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.sampled_softmax_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    seed=None,
    name=&#x27;sampled_softmax_loss&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a faster way to train a softmax classifier over a huge number of
classes.

This operation is for training only.  It is generally an underestimate of
the full softmax loss.

A common use case is to use this method for training, and calculate the full
softmax loss for evaluation or inference as in the following example:

```python
if mode == "train":
  loss = tf.nn.sampled_softmax_loss(
      weights=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      ...)
elif mode == "eval":
  logits = tf.matmul(inputs, tf.transpose(weights))
  logits = tf.nn.bias_add(logits, biases)
  labels_one_hot = tf.one_hot(labels, n_classes)
  loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)
```

See our [Candidate Sampling Algorithms Reference]
(https://www.tensorflow.org/extras/candidate_sampling.pdf)

Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

Note: when doing embedding lookup on `weights` and `bias`, "div" partition
strategy will be used. Support for other partition strategy will be added
later.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`weights`<a id="weights"></a>
</td>
<td>
A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
objects whose concatenation along dimension 0 has shape [num_classes,
dim].  The (possibly-sharded) class embeddings.
</td>
</tr><tr>
<td>
`biases`<a id="biases"></a>
</td>
<td>
A `Tensor` of shape `[num_classes]`.  The class biases.
</td>
</tr><tr>
<td>
`labels`<a id="labels"></a>
</td>
<td>
A `Tensor` of type `int64` and shape `[batch_size, num_true]`. The
target classes.  Note that this format differs from the `labels` argument
of <a href="../../tf/nn/softmax_cross_entropy_with_logits.md"><code>nn.softmax_cross_entropy_with_logits</code></a>.
</td>
</tr><tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A `Tensor` of shape `[batch_size, dim]`.  The forward activations of
the input network.
</td>
</tr><tr>
<td>
`num_sampled`<a id="num_sampled"></a>
</td>
<td>
An `int`.  The number of classes to randomly sample per batch.
</td>
</tr><tr>
<td>
`num_classes`<a id="num_classes"></a>
</td>
<td>
An `int`. The number of possible classes.
</td>
</tr><tr>
<td>
`num_true`<a id="num_true"></a>
</td>
<td>
An `int`.  The number of target classes per training example.
</td>
</tr><tr>
<td>
`sampled_values`<a id="sampled_values"></a>
</td>
<td>
a tuple of (`sampled_candidates`, `true_expected_count`,
`sampled_expected_count`) returned by a `*_candidate_sampler` function.
(if None, we default to `log_uniform_candidate_sampler`)
</td>
</tr><tr>
<td>
`remove_accidental_hits`<a id="remove_accidental_hits"></a>
</td>
<td>
 A `bool`.  whether to remove "accidental hits"
where a sampled class equals one of the target classes.  Default is True.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
random seed for candidate sampling. Default to None, which doesn't set
the op-level random seed for candidate sampling.
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
A `batch_size` 1-D tensor of per-example sampled softmax losses.
</td>
</tr>

</table>

