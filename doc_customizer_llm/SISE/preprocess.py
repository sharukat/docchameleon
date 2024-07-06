
import re
import spacy
import stanza
stanza.download('zh', processors='tokenize,pos')
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

code_regex = [
    "[A-Z][a-zA-Z]+ ?<[A-Z][a-zA-Z]*>",
    "[a-zA-Z0-9\.]+[(][a-zA-Z_,\.]*[)]",
    "(https?://)?[a-zA-Z_\\-/]{2,}(\.[a-zA-Z_0-9\\-]{2,})+[^\s\<\>{\(\),'\"”’}:]*",
    "([\.]?[/]?\w+\.\w+\.?\w+(?:\.\w+)*)",
    "[A-Za-z]+\.[A-Z]+",
    "[@][a-zA-Z]+",
    "(?:\s|^)([a-zA-z]{3,}\.[A-Za-z]+_[a-zA-Z_]+)",
    "\b([A-Z]{2,})\b",
    "(?:\s|^)([A-Z]+_[A-Z0-9_]+)",
    "(?:\s|^)([a-z]+_[a-z0-9_]+)",
    "\w{3,}:\w+[a-zA-Z0-9:]*",
    "(?:\s|^)([a-z]+[A-Z][a-zA-Z]+)(\s|,|\.|\))",
    "(?:\s|^)([A-Z]+[a-z0-9]+[A-Z][a-z0-9]+\w*)(\s|\.\s|\.$|$|,\s)",
    "(?:\s|^)([A-Z]{3,}[a-z0-9]{2,}\w*)(\s|\.\s|\.$|$|,\s)",
    "(?:\s|^)([a-z0-9]+[A-Z]+\w*)(\s|\.\s|\.$|$|,\s)",
    "(?:\s|^)(\w+\([^)]*\))(\s|\.\s|\.$|$|,\s)",
    "([A-Z][a-z]+[A-Z][a-zA-Z]+)(\s|,|\.|\))",
    "([a-z]+[A-Z][a-zA-Z]+)(\s|,|\.|\))",
    "([a-z] )([A-Z][a-z]{3,11})( )",
    "</?[a-zA-Z0-9 ]+>",
    "\{\{[^\}]*\}\}",
    "\{\%[^\%]*\%\}",
    "/[^/]*/",
    "‘[^’]*’",
    "__[^_]*__",
    "\$[A-Za-z\_]+"
]


def split_to_sentences(text: str):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', download_method=None, verbose=False)
    doc = nlp(text)
    # senteces = [sentence.text for sentence in doc.sentences]
    return doc.sentences

def remove_code_blocks(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find and remove all <pre> tags along with their content
    for pre_tag in soup.find_all('pre'):
        pre_tag.decompose()
    
    text_only = soup.get_text()   
    return text_only

def tag_code_elements(text):
    soup = BeautifulSoup(str(text), 'html.parser')
    for tag in soup.find_all(['code', 'tt']):
        tag.string = "NN"

    plain_text = soup.get_text()
    for pattern in code_regex:
        plain_text = re.sub(pattern, "NN", plain_text)
    return plain_text

def embed_and_prefix_sentences(model, sentences):
    # nlp = spacy.load('en_core_web_sm')
    modified_sentences = []
    api_sent_embed = []

    for sent in sentences:
        sent_text = sent.text.strip()
        if sent_text:
            api_sent_embed.append(model(sent_text).vector)


        # Check if the sentence starts with a verb in the present tense, third person singular (VBZ)
        sent = sent.to_dict()
        if sent[0]["xpos"] == 'VBZ':
            modified_sentence = "this " + sent_text
        # Check if the sentence starts with a present participle or gerund (VBG) followed by a noun (NN)
        elif len(sent_text) > 1 and sent[0]["xpos"] == 'VBG' and sent[1]["xpos"] in ['NN', 'NNS']:
            modified_sentence = "for " + sent_text
        else:
            modified_sentence = sent_text
        modified_sentences.append(modified_sentence)
    modified_text = ' '.join(modified_sentences)


    return api_sent_embed, modified_text

def pos(sentence):
    pos_tags = []
    for token in sentence.to_dict():
        if 'xpos' in token:
            pos_tags.append(token['xpos'])
    # pos_tags = [token.get('xpos') for token in sentence.to_dict()]
    pos_sentence = ' '.join(pos_tags)
    return pos_sentence


# text = """
# <div class="block">Resizable-array implementation of the <tt>List</tt> interface.  Implements
#  all optional list operations, and permits all elements, including
#  <tt>null</tt>.  In addition to implementing the <tt>List</tt> interface,
#  this class provides methods to manipulate the size of the array that is
#  used internally to store the list.  (This class is roughly equivalent to
#  <tt>Vector</tt>, except that it is unsynchronized.)

#  <p>The <tt>size</tt>, <tt>isEmpty</tt>, <tt>get</tt>, <tt>set</tt>,
#  <tt>iterator</tt>, and <tt>listIterator</tt> operations run in constant
#  time.  The <tt>add</tt> operation runs in <i>amortized constant time</i>,
#  that is, adding n elements requires O(n) time.  All of the other operations
#  run in linear time (roughly speaking).  The constant factor is low compared
#  to that for the <tt>LinkedList</tt> implementation.

#  <p>Each <tt>ArrayList</tt> instance has a <i>capacity</i>.  The capacity is
#  the size of the array used to store the elements in the list.  It is always
#  at least as large as the list size.  As elements are added to an ArrayList,
#  its capacity grows automatically.  The details of the growth policy are not
#  specified beyond the fact that adding an element has constant amortized
#  time cost.

#  <p>An application can increase the capacity of an <tt>ArrayList</tt> instance
#  before adding a large number of elements using the <tt>ensureCapacity</tt>
#  operation.  This may reduce the amount of incremental reallocation.

#  <p><strong>Note that this implementation is not synchronized.</strong>
#  If multiple threads access an <tt>ArrayList</tt> instance concurrently,
#  and at least one of the threads modifies the list structurally, it
#  <i>must</i> be synchronized externally.  (A structural modification is
#  any operation that adds or deletes one or more elements, or explicitly
#  resizes the backing array; merely setting the value of an element is not
#  a structural modification.)  This is typically accomplished by
#  synchronizing on some object that naturally encapsulates the list.

#  If no such object exists, the list should be "wrapped" using the
#  <a href="../../java/util/Collections.html#synchronizedList-java.util.List-"><code>Collections.synchronizedList</code></a>
#  method.  This is best done at creation time, to prevent accidental
#  unsynchronized access to the list:<pre>
#    List list = Collections.synchronizedList(new ArrayList(...));</pre>
# <p><a name="fail-fast">
#  The iterators returned by this class's <a href="../../java/util/ArrayList.html#iterator--"><code>iterator</code></a> and
#  <a href="../../java/util/ArrayList.html#listIterator-int-"><code>listIterator</code></a> methods are <em>fail-fast</em>:</a>
#  if the list is structurally modified at any time after the iterator is
#  created, in any way except through the iterator's own
#  <a href="../../java/util/ListIterator.html#remove--"><code>remove</code></a> or
#  <a href="../../java/util/ListIterator.html#add-E-"><code>add</code></a> methods, the iterator will throw a
#  <a href="../../java/util/ConcurrentModificationException.html" title="class in java.util"><code>ConcurrentModificationException</code></a>.  Thus, in the face of
#  concurrent modification, the iterator fails quickly and cleanly, rather
#  than risking arbitrary, non-deterministic behavior at an undetermined
#  time in the future.

#  <p>Note that the fail-fast behavior of an iterator cannot be guaranteed
#  as it is, generally speaking, impossible to make any hard guarantees in the
#  presence of unsynchronized concurrent modification.  Fail-fast iterators
#  throw <code>ConcurrentModificationException</code> on a best-effort basis.
#  Therefore, it would be wrong to write a program that depended on this
#  exception for its correctness:  <i>the fail-fast behavior of iterators
#  should be used only to detect bugs.</i>
# <p>This class is a member of the
#  <a href="../../../technotes/guides/collections/index.html">
#  Java Collections Framework</a>.</p></p></p></p></p></p></p></div>
# """

# sentence = "An application can increase the capacity of an NOUN instance before adding a large number of elements using the NOUN operation."

# body = "<p>Given:</p>\n\n<pre><code>Element[] array = new Element[] { new Element(1), new Element(2), new Element(3) };\n</code></pre>\n\n<p>The simplest answer is to do:</p>\n\n<pre><code>List&lt;Element&gt; list = Arrays.asList(array);\n</code></pre>\n\n<p>This will work fine.  But some caveats:</p>\n\n<ol>\n<li>The list returned from asList has <strong>fixed size</strong>.  So, if you want to be able to add or remove elements from the returned list in your code, you'll need to wrap it in a new <code>ArrayList</code>.  Otherwise you'll get an <code>UnsupportedOperationException</code>.</li>\n<li>The list returned from <code>asList()</code> is backed by the original array.  If you modify the original array, the list will be modified as well.  This may be surprising. </li>\n</ol>\n"
# value = [{'id': 1, 'text': '<', 'upos': 'PUNCT', 'xpos': '-LRB-', 'start_char': 0, 'end_char': 1, 'misc': 'SpaceAfter=No'}, {'id': 2, 'text': 'p>', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'start_char': 1, 'end_char': 3, 'misc': 'SpaceAfter=No'}, {'id': 3, 'text': 'Given', 'upos': 'VERB', 'xpos': 'VBN', 'feats': 'Tense=Past|VerbForm=Part', 'start_char': 3, 'end_char': 8, 'misc': 'SpaceAfter=No'}, {'id': 4, 'text': ':', 'upos': 'PUNCT', 'xpos': ':', 'start_char': 8, 'end_char': 9, 'misc': 'SpaceAfter=No'}, {'id': 5, 'text': '</', 'upos': 'PUNCT', 'xpos': '-LRB-', 'start_char': 9, 'end_char': 11, 'misc': 'SpaceAfter=No'}, {'id': 6, 'text': 'p', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'start_char': 11, 'end_char': 12, 'misc': 'SpaceAfter=No'}, {'id': 7, 'text': '>', 'upos': 'PUNCT', 'xpos': '-RRB-', 'start_char': 12, 'end_char': 13, 'misc': 'SpacesAfter=\\n\\n'}]
# result = pos(value)
# # result = pos(body)
# # for res in result:
# #     res = res.to_dict()
# #     print(res)
# #     print(type(res))
# #     break
# print(result)