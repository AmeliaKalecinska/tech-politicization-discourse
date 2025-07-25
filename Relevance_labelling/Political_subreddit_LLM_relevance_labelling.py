import os
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="")

def build_prompt(passage: str) -> str:
    return f"""You are an Relevance Analyst working alongside software developers. You analyze posts from social media to identify tech-company politicization. 
    The social media application is Reddit and all posts are taken from politics-oriented subreddits.
    Your task is to classify whether a Reddit post discusses the issue of the voluntary politicization of tech companies and CEOs using binary classification. 
    You will be given a passage containing the title and optional selftext.
    Definition:
    The politicization of tech companies refers to instances where a technology company appears to voluntarily align itself with a specific political ideology, 
    agenda, or movement. This includes situations where the company is perceived as selectively enforcing policies, censoring content, amplifying certain political 
    messages, suppressing opposing views, or making public political statements or decisions that reflect partisan leanings. 
    The focus is on the voluntary and proactive stance of the company or CEO, rather than actions resulting from external political or regulatory pressure. 
    The politicization may manifest through product design choices, moderation strategies, public communications, or corporate affiliations, and raises
    concerns about bias, fairness, and the influence of corporate power on democratic discourse.
    Tech companies and CEOs often associated with politicization discussions include:
    •⁠  ⁠Meta (Facebook, Instagram), and CEO Mark Zuckerberg
    •⁠  ⁠Google and its executives (Sundar Pichai, YouTube's leadership)
    •⁠  ⁠Twitter/X and Elon Musk
    •⁠  ⁠Amazon and Jeff Bezos
    •⁠  ⁠Apple and Tim Cook
    •⁠  Microsoft and Satya Nadella
    •⁠  OpenAI⁠ and Sam Altman
    
    Assign:
    •⁠  ⁠Return label 1 (Politicization) if the post makes claims (explicitly or implicitly) that a tech company or CEO has knowingly and voluntarily engaged in political bias against or in favor of a certain party, political ideology, political censorship, selective enforcement, or partisan messaging.
    •⁠  ⁠Return label 1 (Politicization) if the post claims that a company or CEO is taking a stance on issues that are heavily politicized such as climate change or vaccine mandates.
    •  Return label 1 (Politicization) even if the post does not make a claim, but only reports on a situation, an action or lack of action that would imply political bias, in such situations use your background knowledge to determine this, this is particularly relevant for selectively chosing to act in certain cases and not others.
    •  Return label 1 (Politicization) in cases of tech company executives making a commentary on the government or governmental institutions.
    •⁠  ⁠Return label 0 (Not Politicization) if the post discusses regulatory actions, general ethical issues, or tech-related political debates without alleging that the company is taking sides.
    •⁠  ⁠Return label 0 (Not Politicization) if the post reports on a company or CEO being forced or pressured to collabrate with the government, or was used for a policital agenda without the knowledge of said tech company.
    •⁠  ⁠Always include a short rationale explaining why you assigned the label.
    
    Examples:
    Post: "A group of Obama veterans are banding together to invest in tech that can help Democrats win"
    Label: 0
    Rationale: External political debate, no voluntary alignment by a tech company.
    
    Post: “Amazon Studios Blasted for Censoring Scenes in Christmas Classic Its A Wonderful Life”
    Label: 0
    Rationale: The Post mentions how Amazon is censoring a movie but censorship purely is not relevant to our topic since we lack context to conclude that this particular censorship is in favor of some political party to conclude the politicization of Amazon
    
    Post: "Elon Musk Will Fund Twitter Deal With Money From Countries That Suppress Free Speech"
    Label: 1
    Rationale: Voluntary partnership with politically repressive governments aligns with definition of politicization.
    
    Post: "Facebook demands academics disable tool showing who is being targeted by political ads"
    Label: 1
    Rationale: The post suggests Facebook is selectively controlling access to information about political ad targeting, implying potential bias in how political content is moderated or displayed.
    
    Now classify the following Reddit post:
    Passage: {passage}
    Label:
    Rationale:
"""

def classify_politicization(passage: str) -> tuple:
    prompt = build_prompt(passage)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = response.choices[0].message.content.strip()

        label_match = re.search(r"Label:\s*([01])", output)
        label = label_match.group(1) if label_match else "error"

        rationale_lines = [line for line in output.splitlines() if not line.strip().lower().startswith("label:")]
        rationale = "\n".join(rationale_lines).strip()

        return label, rationale
    except Exception as e:
        return "error", f"API Error: {str(e)}"

def process_excel(input_path: str, output_path: str):
    df = pd.read_excel(input_path)
    labels = []
    rationales = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        passage = f"{row['title']} {row['selftext']}"
        label, rationale = classify_politicization(passage)
        labels.append(label)
        rationales.append(rationale)

    if "llm_label" in df.columns:
        df.drop(columns=["llm_label"], inplace=True)
    if "rationale" in df.columns:
        df.drop(columns=["rationale"], inplace=True)

    df.insert(11, "llm_label", labels)
    df.insert(12, "rationale", rationales)

    df.to_excel(output_path, index=False)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    input_xlsx_file = ""
    output_xlsx_file = ""
    process_excel(input_xlsx_file, output_xlsx_file)
