from pathlib import Path
import glob
import json
import os
import shutil
import lib.global_settings as s

# class MarkdownFileNotFoundError(Exception):
#     pass

# def search_and_read_markdown(folder_path, file_name):
#     if not file_name:
#       folder_path = Path(folder_path)
#       files = folder_path.rglob(f"{file_name}.md")

#       matching_files = list(files)
#       if not matching_files:
#           raise MarkdownFileNotFoundError(f"File '{file_name}' not found in the specified folder and its subdirectories.")

#       # Assuming there's only one matching file, read its content
#       matching_file = matching_files[0]
#       if matching_file.suffix.lower() == ".md":
#           return matching_file.read_text()

#       # If the file is not a markdown file, raise an exception
#       raise MarkdownFileNotFoundError(f"File '{file_name}' is not a markdown file.")

def read_markdown_file(filename):
  """Reads a Markdown file and returns its contents as a string."""

  filename_with_ext = os.path.join(s.DOCUMENTATION_PATH, filename) + ".md"  # Add the .md extension
  try:
    with open(filename_with_ext, "r", encoding="utf-8") as file:
      content = file.read()
      return content
  except FileNotFoundError:
    # print(f"Error: File '{filename_with_ext}' not found.")
    return None



def prompt_task(type: str):

  with open(os.path.join(s.ASSETS_PATH, 'issue_types_tasks.json'), 'r') as f:
    tasks = json.load(f)
  return tasks.get(type)



# Move files and rename based on the folder structure
def move_files(main_dir, target_dir):
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith('.md'):
                # Construct the new filename based on the directory structure
                rel_dir = os.path.relpath(root, os.path.dirname(main_dir))
                new_filename = f"{rel_dir}.{file}".replace(os.sep, '.')

                # Construct the full paths for the source and destination files
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, new_filename)

                # Move the file
                shutil.move(src_file, dst_file)
