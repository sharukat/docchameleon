from mdutils.mdutils import MdUtils
from mdutils import Html
from lib.common import stub
import modal


def return_template1(api_name, prefix, imports, code, urls):
    mdFile = MdUtils(file_name=f"{api_name}-Customized")
    mdFile.new_header(level=1, title='Customized Content')
    mdFile.new_paragraph(str(prefix))
    mdFile.insert_code(f"{imports} \n{code}")
    mdFile.new_header(level=2, title="Additional Resources")
    if urls:
        mdFile.new_line('  - ' + mdFile.new_inline_link(link=urls[0]))
    output = mdFile.get_md_text()
    return output

def return_template2(api_name, solution, urls):
    mdFile = MdUtils(file_name=f"{api_name}-Customized")
    mdFile.new_header(level=1, title='Customized Content')
    mdFile.new_paragraph(str(solution))
    mdFile.new_header(level=2, title="Additional Resources")
    if urls:
        mdFile.new_line('  - ' + mdFile.new_inline_link(link=urls[0]))
    output = mdFile.get_md_text()
    return output

def return_template3(api_name, prefix, imports, code):
    mdFile = MdUtils(file_name=f"{api_name}-Customized")
    mdFile.new_header(level=2, title="Additional Resources")
    # mdFile.new_inline_link(link=urls[0])
    output = mdFile.get_md_text()
    return output