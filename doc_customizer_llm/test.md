# Stack Overflow FAQs with Accepted Answers

### Tensor has shape [?, 0] -- how to reshape to [?,]


I think you should use [`tf.not_equal`](https://www.tensorflow.org/api_docs/python/tf/not_equal) to perform elementwise comparison on the tensor.



```
src = tf.constant([0, 1, 1, 0], dtype=tf.int8)
tf.gather(src, tf.where(tf.not_equal(src, 0))).eval(session=tf.Session())

array([[1],
       [1]], dtype=int8)

```

You can also shorten this a bit and use [`tf.boolean_mask`](https://www.tensorflow.org/api_docs/python/tf/boolean_mask) instead of `tf.where` and `tf.gather`:



```
tf.boolean_mask(src, tf.not_equal(src, 0)).eval(session=tf.Session())
array([1, 1], dtype=int8)

```

Note the difference in the shape of the outputs.
