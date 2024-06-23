from mdutils.mdutils import MdUtils
from mdutils import Html
from lib.common import stub
import modal


def return_template1(api_name, prefix, imports, code, urls):
    mdFile = MdUtils(file_name=f"{api_name}-Customized")
    mdFile.new_header(level=1, title='Customized Content')
    mdFile.new_header(level=2, title='Usage Example based on Stack Overflow Question')
    mdFile.new_paragraph(str(prefix))
    mdFile.insert_code(f"{imports} \n{code}")
    if urls:
        mdFile.new_header(level=2, title="Related Stack Overflow Posts with Accepted Answers")
        for i, url in enumerate(urls):
            label = f'Link {i}'
            mdFile.new_line('  - ' + mdFile.new_inline_link(link=url, text=label))

    # if course_urls:
    #     mdFile.new_header(level=3, title="Related Online Courses")
    #     for i, url in enumerate(course_urls):
    #         label = f'Link {i}'
    #         mdFile.new_line('  - ' + mdFile.new_inline_link(link=url, text=label))


    output = mdFile.get_md_text()
    return output

def return_template2(api_name, solution, urls):
    mdFile = MdUtils(file_name=f"{api_name}-Customized")
    mdFile.new_header(level=1, title='Customized Content')
    mdFile.new_header(level=2, title='API Behavior Clarification based on Stack Overflow Question')
    mdFile.new_paragraph(str(solution))
    if urls:
        mdFile.new_header(level=2, title="Related Stack Overflow Posts with Accepted Answers")
        for i, url in enumerate(urls):
            label = f'Link {i}'
            mdFile.new_line('  - ' + mdFile.new_inline_link(link=url, text=label))

    # if course_urls:
    #     mdFile.new_header(level=3, title="Related Online Courses")
    #     for i, url in enumerate(course_urls):
    #         label = f'Link {i}'
    #         mdFile.new_line('  - ' + mdFile.new_inline_link(link=url, text=label))

    output = mdFile.get_md_text()
    return output

def return_template3(api_name, urls, course_urls):
    mdFile = MdUtils(file_name=f"{api_name}-Customized")
    mdFile.new_header(level=2, title="Additional Resources")
    if urls:
        mdFile.new_header(level=3, title="Related Stack Overflow Posts with Accepted Answers")
        for url in urls:
            mdFile.new_line('  - ' + mdFile.new_inline_link(link=url))

    if course_urls:
        mdFile.new_header(level=3, title="Related Online Courses")
        for url in course_urls:
            mdFile.new_line('  - ' + mdFile.new_inline_link(link=url))
    output = mdFile.get_md_text()
    return output