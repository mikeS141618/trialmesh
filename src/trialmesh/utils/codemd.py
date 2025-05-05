import os
import sys

def read_file(path):
    """Read a file and return its contents.

    Args:
        path: Path to the file to read

    Returns:
        The file contents as a string, or an error message
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Could not read {path}: {e}"

def get_src_root():
    """Get the root of the source directory.

    Returns:
        Path to the src directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    return src_dir

def get_project_root():
    """Get the root of the project directory.

    Returns:
        Path to the project root directory (parent of src)
    """
    return os.path.abspath(os.path.join(get_src_root(), '..'))

def build_tree(root):
    """Build a text representation of the directory tree.

    Args:
        root: Root directory to build tree from

    Returns:
        String representation of the directory tree
    """
    tree_lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_path = os.path.relpath(dirpath, root)
        indent = '    ' * (0 if rel_path == '.' else rel_path.count(os.sep))
        base = '.' if rel_path == '.' else rel_path
        tree_lines.append(f"{indent}{os.path.basename(base)}" if base != '.' else f"{indent}{os.path.basename(root)}")
        for fname in sorted(filenames):
            tree_lines.append(f"{indent}    {fname}")
    return '\n'.join(tree_lines)

def collect_python_files(root):
    """Collect all Python files in a directory recursively.

    Args:
        root: Root directory to search from

    Returns:
        List of tuples (relative_path, file_content) for each Python file
    """
    py_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith('.py'):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, os.path.dirname(root))
                py_files.append((rel_path, read_file(full_path)))
    return py_files

def generate_codemd():
    """Generate a Markdown file with code documentation.

    This function creates a comprehensive Markdown document that includes:
    1. The project README
    2. pyproject.toml configuration
    3. Directory structure
    4. All Python source files with content

    The output file is written to codecomplete.md in the project root.
    """
    src_dir = get_src_root()
    project_root = get_project_root()
    readme_path = os.path.join(project_root, 'README.md')
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    output_path = os.path.join(project_root, 'codecomplete.md')

    readme = read_file(readme_path)
    pyproject = read_file(pyproject_path)
    tree = build_tree(src_dir)
    py_files = collect_python_files(src_dir)

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("# Project README\n\n")
        out.write(f"```\n{readme}\n```\n\n")

        out.write("# pyproject.toml\n\n")
        out.write(f"```toml\n{pyproject}\n```\n\n")

        out.write("# src Directory Tree\n\n")
        out.write(f"```\n{tree}\n```\n\n")

        out.write("# Python Source Files\n\n")
        for rel_path, content in py_files:
            out.write(f"## {rel_path}\n\n")
            out.write(f"```python\n{content}\n```\n\n")

def cli_main():
    """Entry point for trialmesh-codemd CLI tool."""
    generate_codemd()
    print("codecomplete.md generated at project root.")

if __name__ == "__main__":
    cli_main()