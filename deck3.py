import os

files = [
    "0-intro",
    "1-simulation",
    "2-data",
    "3-alphas_betas",
    "4-visualization",
    "5-portfolios",
    "6-autoregression",
    "7-linear",
    "8-trees",
    "9-classification",
    "10-nets"
]
for file in files:
    os.system(f'decktape automatic docs\{file}.html docs\pdfs\{file}.pdf')

os.system("git add .")
os.system("git commit -m 'message'")
os.system("git push origin main")