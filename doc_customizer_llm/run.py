import requests
import json
endpoint = "https://sharukat--code-langchain-serve-dev.modal.run/codelangchain/invoke"

input = {
    "title":"TensorFlow while loop with condition dependent on body",
    "question":"""
        <p>I want to have a while loop with the condition dependent on a tensor computed in the loop body, but I don't know how to accomplish this with <a href="https://www.tensorflow.org/api_docs/python/tf/while_loop" rel="nofollow noreferrer"><code>tf.while_loop()</code></a>.</p>

    <p>My input processing includes random cropping, but some crops can lead to low-quality examples and I want to discard those and try a new random crop until an example of sufficient quality is obtained. The inputs are cropped by</p>

    <pre><code>import numpy as np
    import tensorflow as tf
    IMAGE_SHAPE = [960, 720]
    CROP_SHAPE = [320, 240]
    max_begin_index = np.array(IMAGE_SHAPE) - np.array(CROP_SHAPE)
    crop_begin_index = tf.round(tf.random_uniform([2]) * max_begin_index)
    img_crop = tf.slice(img, crop_begin_index, crop_shape + [-1])
    </code></pre>

    <p>and the condition is</p>

    <pre><code>cond = tf.count_nonzero(img_crop &gt; 0) &gt; 0.5 * tf.size(img_crop)
    </code></pre>

    <p>Going over the documentation and examples of <code>tf.while_loop(cond, body, loop_vars, ...)</code>, what I understand is that both <code>cond</code> and <code>body</code> should take the same arguments given in <code>loop_vars</code>.
    I don't see how I can have <code>cond</code> depend on <code>img_crop</code> which would be calculated inside <code>body</code>, and isn't provided in <code>loop_vars</code>.</p>

    <p>I could equivalently compute <code>cond</code> using <code>crop_begin_index</code> without actually cropping, but it depends on the random values computed inside the loop, so I have the same problem.</p>

    <p>Is this indeed a limitation of TF looping? If not, how can I rewrite my code to use <code>tf.while_loop()</code>?</p>
    """,
    "api_name":"tf.while_loop",
    "issue_type":"Documentation Replication on Other Examples",
    "ground_truth":"""
    <p>The arguments that are passed on to the <code>condition</code> function are the arguments returned from your <code>body</code> function. So you just have to return that value that you want to base your condition on in the <code>body</code> function, then carry out the condition on that value in your <code>cond</code> function. Something like, </p>

    <pre><code>def body(image_shape, crop_shape, img_crop):
        max_begin_index = np.array(IMAGE_SHAPE) - np.array(CROP_SHAPE)
        crop_begin_index = tf.round(tf.random_uniform([2]) * max_begin_index)
        img_crop = tf.slice(img, crop_begin_index, crop_shape + [-1])
        return (image_shape, crop_shape, img_crop)

    def cond(image_shape, crop_shape, img_crop):
        return tf.count_nonzero(img_crop &gt; 0) &gt; 0.5 * tf.size(img_crop)

    image_shape, crop_shape, img_crop = tf.while_loop(cond=cond, body=body, loop_vars=([960, 720], [320, 240], img_crop))
    </code></pre>

    <p>Don't have access to an interpreter right now, so there might be some syntax problems there, but something like that. </p>

    <p>Also, if I recall correctly, the body and the condition need to be pure functions, you cannot alter the outer state from within the functions.</p>

    <p>Also note, you'll need to specify some initial value for <code>img_crop</code> in the loop vars.</p>

    <p>Moreover, by default, <code>tf.while_loop</code> expects the shapes of all the <code>loop_vars</code> to remain the same across all loop runs. You can modify this through the <code>shape_invariants</code> argument. </p>
    """,
}

data = {
    "input": str(input),
    "config": {"recursion_limit": 50},
    "kwargs": {},
}

r = requests.post(
    url=endpoint,
    json=data,
)

print(r.text)