import os
import sys

file = sys.argv[1]
os.system(f'decktape automatic docs\{file}.html docs\pdfs\{file}.pdf')
os.system("git add .")
os.system("git commit -m 'message'")
os.system("git push origin main")