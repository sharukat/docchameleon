<table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A Tensor of rank at least 3. Must be of type `float16`, `float32`, or
`float64`.
</td>
</tr><tr>
<td>
`filters`<a id="filters"></a>
</td>
<td>
A Tensor of rank at least 3.  Must have the same type as `input`.
</td>
</tr><tr>
<td>
`stride`<a id="stride"></a>
</td>
<td>
An int or list of `ints` that has length `1` or `3`.  The number of
entries by which the filter is moved right at each step.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
'SAME' or 'VALID'. See
[here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
for more information.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
An optional `string` from `"NWC", "NCW"`.  Defaults to `"NWC"`,
the data is stored in the order of
`batch_shape + [in_width, in_channels]`.  The `"NCW"` format stores data
as `batch_shape + [in_channels, in_width]`.
</td>
</tr><tr>
<td>
`dilations`<a id="dilations"></a>
</td>
<td>
An int or list of `ints` that has length `1` or `3` which
defaults to 1. The dilation factor for each dimension of input. If set to
k > 1, there will be k-1 skipped cells between each filter element on that
dimension. Dilations in the batch and depth dimensions must be 1.
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