from claude_summarizer import ai_question_summarizer

# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================

title = "TF1 to TF2 migration"
body = """"
    <p>Hello I am new to tensorflow and I am working on a code that I would like to migrate from tensorflow 1 to 2. I have this line of code:</p>
    <pre><code>x1 = tf.compat.v1.placeholder(tf.float32, [], name=&quot;x1&quot;)
    </code></pre>
    <p>As mentioned in <a href="https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder" rel="nofollow noreferrer">https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder</a>, I should use <code>keras.Input</code>. But even when specifying the shape, I can't have the same tensor as with compat.v1:</p>
    <pre><code>x2 = tf.keras.Input(shape=[], dtype=tf.float32, name=&quot;x2&quot;)
    </code></pre>
    <p>To check the shape I use <code>tf.shape(x1)</code> or <code>tf.shape(x2)</code>, but the shapes are not the same. Could anyone explain to me how to have, in TF2, the same shape as in TF1 ?
    Thanks and regards</p>
"""

response = ai_question_summarizer(title, body)
print(response)