### Document 1:

<p>You can use just <code>.decode(&quot;utf-8&quot;)</code> function on bytes object, that you get after apply <code>.numpy()</code> method for tensor</p>
----------------------------------------------------------------------------------------------------

### Document 2:

<p>I'm assuming you need the <code>filepath</code> as a string so you can load the <code>.wav</code> files as some 16-bit float to feed into a model. To avoid the performance downsides of <code>tf.py_function</code>, it's probably best to try to make the best of relevant parts of the tensorflow API, most of which support <code>Tensor</code> as inputs.</p>
<p>If, for example, your dataset consisted of images, you might want to do something like:</p>
<pre class="lang-py prettyprint-override"><code>def path2img(path):
    img_raw = tf.io.read_file(path)
    return tf.io.decode_image(img_raw, 3)

dataset = tf.data.Dataset.list_files(PATH + &quot;*.png&quot;)
dataset = dataset.map(path2img)
</code></pre>
<p>for <code>.wav</code> files, try:</p>
<pre><code>def path2wav(path):
    audio_raw = tf.io.read_file(path)
    return tf.audio.decode_wav(audio_raw)
    </code>
</pre>
----------------------------------------------------------------------------------------------------

### Document 3:

<p><strong>UPDATE</strong> for TF 2</p>

<p>The above solution will not work with TF 2 (tested with 2.2.0), even when replacing <code>tf.py_func</code> with <code>tf.py_function</code>, giving</p>
...

<h2>Edit:</h2>

<p>As you pointed out in the comment, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator" rel="noreferrer"><code>tf.data.Dataset.from_generator()</code></a> has a third parameter which sets the shape of the output tensor, so instead of <code>feature.set_shape()</code> just pass the shape as <code>output_shapes</code> in <code>from_generator()</code>.</p>