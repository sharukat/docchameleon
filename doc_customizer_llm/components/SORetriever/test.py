from so_retrieval import retrieve_relevant_from_so

# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================

title = "Custom initializer for get_variable"
body = '''
<p>How can one specify a custom initializer as the third argument for <code>tf.get_variable()</code>? Specifically, I have a variable <code>y</code> which I want to initialize using another (already initialized) variable <code>x</code>. </p>

<p>This is easy to do using <code>tf.Variable()</code>, just say, <code>y = tf.Variable(x.initialized_value())</code>. But I couldn't find an analog in the documentation for <code>tf.get_variable()</code>.</p>


'''

docs = retrieve_relevant_from_so(question_title=title, question_body=body)
if docs != None:
  for idx, doc in enumerate(docs):
    print(f"{'=' * 100}")
    print(f"Document Rank: {idx + 1} | Relevant Score: {doc.metadata['relevance_score']}")
    print(f"{'=' * 100}\n")
    print(f"{doc.page_content}")
    print(f"\nURL: {doc.metadata['URL']}\n\n")
else:
  print("No relevant posts found on Stack Overflow.")