import os
import shutil

latex_dir = 'latex'
docs_dir = 'docs'

# Get list of files in latex directory
files = os.listdir(latex_dir)
files = [f for f in files if os.path.isfile(os.path.join(latex_dir, f))]

# Delete files that aren't .tex or .pdf
for file in files:
    if not file.endswith(('.tex', '.pdf', '.qmd')):
        os.remove(os.path.join(latex_dir, file))

# Copy PDF files to docs directory
for file in files:
    if file.endswith('.pdf'):
        shutil.copy2(os.path.join(latex_dir, file), os.path.join(docs_dir, file))
