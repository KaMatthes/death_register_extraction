from openai import OpenAI
import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

# nur für status bar
tqdm.pandas()


client = OpenAI(
    base_url="https://gpustack.unibe.ch/v1-openai",
    api_key="gpustack_a0f67c08841e32f3_04262ea3cb3172504ea071b791d9ea38"
)

model = "gpt-oss-120b"

#region llm promts
def extract_all(text1, text2, text3):

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Du bist ein Informationsextraktionssystem für historische Sterberegister."},
                {"role": "user", "content": f"""
Todesort/Ursache:
{text1}

Name/Beruf/Familie:
{text2}

Wohnort/Geburtsdatum:
{text3}

Gib nur JSON zurück mit folgenden Feldern:
Todesort, Strasse_Todesort, Hausnummer_Todesort, Todesursachen,
Name, Beruf, Vater, Mutter, Zivilstand, Religion, Heimatort,
Wohnort, Strasse_Wohnort, Hausnummer_Wohnort, Geburtsdatum
"""}
            ]
        )

        content = response.choices[0].message.content
        content = content.replace("```json","").replace("```","").strip()

        data = json.loads(content)
        expected_keys = [
"Todesort","Strasse_Todesort","Hausnummer_Todesort","Todesursachen",
"Name","Beruf","Vater","Mutter","Zivilstand","Religion","Heimatort",
"Wohnort","Strasse_Wohnort","Hausnummer_Wohnort","Geburtsdatum"
]

        for k in expected_keys:
            if k not in data:
                data[k] = "-" if k != "Todesursachen" else []

        # sicherstellen dass Todesursachen immer Liste ist
        if not isinstance(data.get("Todesursachen"), list):
            if data.get("Todesursachen") in [None, "-", ""]:
                data["Todesursachen"] = []
            else:
                data["Todesursachen"] = [data["Todesursachen"]]

        return data

    except Exception as e:
        print("LLM Fehler:", e)

        return {
            "Todesort": "-",
            "Strasse_Todesort": "-",
            "Hausnummer_Todesort": "-",
            "Todesursachen": [],
            "Name": "-",
            "Beruf": "-",
            "Vater": "-",
            "Mutter": "-",
            "Zivilstand": "-",
            "Religion": "-",
            "Heimatort": "-",
            "Wohnort": "-",
            "Strasse_Wohnort": "-",
            "Hausnummer_Wohnort": "-",
            "Geburtsdatum": "-"
        }
    #endregion



#region helper functions
'''
def clean_last_col(df): #TODO: noch nötig??
    # alles weg vor "wohnhaft"
    df["Wohnort/Geburtsdatum"] = df["Wohnort/Geburtsdatum"].astype(str).apply(
        lambda x: re.sub(r"^.*?(wohnhaft)", r"\1", x, flags=re.IGNORECASE).strip()
    )
    return df
'''

def remove_laut(df):
    df["Todesort/Ursache"] = df["Todesort/Ursache"].astype(str).apply(
        lambda x: re.sub(r"\blaut\b", "", x, flags=re.IGNORECASE).strip()
    )
    return df
#endregion



#region main program
def main_llm(df):

    df = remove_laut(df)

    results = df.progress_apply(
        lambda row: extract_all(
            row["Todesort/Ursache"],
            row["Name/Beruf/Vater,Mutter/Zivilstand/Wohn/Heimatort/Konfession"],
            row["Wohnort/Geburtsdatum"]
        ),
        axis=1
    )

    results = results.tolist()

    df["Todesort"] = [r["Todesort"] for r in results]
    df["Strasse/Institution (Todesort)"] = [r["Strasse_Todesort"] for r in results]
    df["Hausnummer (Todesort)"] = [r["Hausnummer_Todesort"] for r in results]
    df["Todesursachen"] = [
    "; ".join(r["Todesursachen"]) if isinstance(r["Todesursachen"], list)
    else str(r["Todesursachen"])
    for r in results
    ]   

    df["Name"] = [r["Name"] for r in results]
    df["Beruf"] = [r["Beruf"] for r in results]
    df["Vater"] = [r["Vater"] for r in results]
    df["Mutter"] = [r["Mutter"] for r in results]
    df["Zivilstand"] = [r["Zivilstand"] for r in results]
    df["Religion"] = [r["Religion"] for r in results]
    df["Heimatort"] = [r["Heimatort"] for r in results]

    df["Wohnort"] = [r["Wohnort"] for r in results]
    df["Strasse (Wohnort)"] = [r["Strasse_Wohnort"] for r in results]
    df["Hausnummer (Wohnort)"] = [r["Hausnummer_Wohnort"] for r in results]
    df["Geburtsdatum"] = [r["Geburtsdatum"] for r in results]

    return df
#--------------------------------------------------------------------------------------------------------
# Reading out files 
#--------------------------------------------------------------------------------------------------------
# folder of files:
'''
folderInput = Path(__file__).parent / "data" / "neues_Format" #path to folder with files
folderOutput = Path(__file__).parent / "data" / "neues_Format" / "output_dates" #path to folder for output files
for file in folderInput.glob("*.csv"):
    #print("File: ", file.name)

    df = pd.read_csv(file)
    df = main_llm(df)

    output_path = folderOutput / ("gpt-oss_" + file.name)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
'''

# single file:
'''
file = Path(__file__).parent / "1890_1892_Hottingen_28-46.csv" #path to file
folderOutput = Path(__file__).parent / "data" / "neues_Format" / "output_llm" #path to folder for output files
df = pd.read_csv(file) #read the csv file
df = main_llm(df) 
output_path = folderOutput / ("gpt-oss_" + file.name) #path + name for output file
df.to_csv(output_path, index=False, encoding="utf-8-sig") #save
'''

#endregion





#--------------------------------------------------------------------------------------------------------
# testing
#--------------------------------------------------------------------------------------------------------

#todesort_ursache_text = "ist gestorben zu Heinrich in der Kantonalstrafanstalt Bettenbach an Pleuritis, laut"
#namen_beruf_leben_text = ". Scholler, Susanna geb. Altorfer, Beruf: Tochter des Heinrich Altorfer und der Wittwe des Joseph Scholler, Civilstand: von Belfort, Frankreich,,"
#wohnort_geburtstag_text = "wohnhaft in Aussersihl, geboren den zweiten Dezember achtzehnhundert achtzig."

#print(extract_todesort_ursache(todesort_ursache_text))
#print(extract_namen_beruf_leben(namen_beruf_leben_text))
#print(extract_wohnort_geburtstag(wohnort_geburtstag_text))


file = Path(__file__).parent / "Qwent" / "1886_1899_Altstetten_qwen_extraktion_strukturiert.csv"
df = pd.read_csv(file) 
#df = df.loc[[1, 2]]
#df = df.loc[100:104]
#df = df.sample(n=1)
df = main_llm(df) 
output_path = Path(__file__).parent / ("gpt-oss_" + file.name) 
df.to_csv(output_path, index=False, encoding="utf-8-sig") 
