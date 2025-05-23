import json
import os
import shutil
import lib.global_settings as s


def read_markdown_file(filename):
    """Reads a Markdown file and returns its contents as a string."""

    filename_with_ext = os.path.join(s.DOCUMENTATION_PATH, filename) + ".md"
    try:
        with open(filename_with_ext, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return None


def prompt_task(type: str):
    with open(os.path.join(s.ASSETS_PATH, "issue_types_tasks.json"), "r") as f:
        tasks = json.load(f)
    return tasks.get(type)


# Move files and rename based on the folder structure
def move_files(main_dir, target_dir):
    for root, _, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".md"):
                # Construct the new filename based on the directory structure
                rel_dir = os.path.relpath(root, os.path.dirname(main_dir))
                new_filename = f"{rel_dir}.{file}".replace(os.sep, ".")

                # Construct the full paths for the source and destination files
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, new_filename)

                # Move the file
                shutil.move(src_file, dst_file)
