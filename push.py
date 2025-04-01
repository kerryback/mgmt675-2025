import os
os.system("git pull origin main")
with open("docs/CNAME", "w") as f:
    f.write("https://mgmt675-2025.kerryback.com")
os.system("git add .")
os.system("git commit -m 'update'")
os.system("git push origin main")