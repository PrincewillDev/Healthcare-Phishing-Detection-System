import pandas as pd
df = pd.read_csv("data/raw/healthcare_phishing.csv")
phishing_sample = df[df["Email Type"] == "Phishing Email"].sample(15, random_state=42)
for text in phishing_sample["Email Text"]:
    print(text[:5000])
    print("---")
