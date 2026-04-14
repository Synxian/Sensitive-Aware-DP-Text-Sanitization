import json
import os
import pandas as pd

data = []
file_path = "mimic_s.json"
with open(file_path, "r") as f:
    json_data = json.load(f)
for doc in json_data:
    data.append(
        {
            "text_id": doc["id"],
            "text": doc["output"],
        }
    )
df = pd.DataFrame(data)
df.to_csv("mimic.csv", index=False)
