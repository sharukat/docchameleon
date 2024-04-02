from mdutils.mdutils import MdUtils
from mdutils import Html
from lib.common import stub
import modal

# nfs = modal.NetworkFileSystem.from_name("my-nfs", create_if_missing=True)

# @stub.function(network_file_systems={"/root/markdown": nfs})
def output_template(api_name, prefix, imports, code, urls):
    mdFile = MdUtils(file_name=f"{api_name}-Customized")
    mdFile.new_header(level=1, title='Customized Content')
    mdFile.new_paragraph(str(prefix))
    mdFile.insert_code(f"{imports} \n{code}")
    mdFile.new_header(level=2, title="Additional Resources")
    mdFile.new_inline_link(link=urls[0])
    output = mdFile.get_md_text()
    return output