import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import re
import signal
import sys

client = OpenAI(api_key="")

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("API call exceeded timeout limit.")

signal.signal(signal.SIGALRM, handler)

NEW_PROMPT = '''
You are a Political Analyst working alongside software developers. You analyze posts from social media to identify tech-company politicization. The social media application is Reddit and all posts are taken from tech-oriented subreddits.

Your task is to classify whether a Reddit post discusses the issue of the voluntary politicization of tech companies and CEOs using binary classification. You will be given a passage containing the title and optional selftext.

Definition:The politicization of tech companies refers to instances where a technology company appears to voluntarily align itself with a specific political ideology, agenda, or movement. This includes situations where the company is perceived as selectively enforcing policies, censoring content, amplifying certain political messages, suppressing opposing views, or making public political statements or decisions that reflect partisan leanings. The focus is on the voluntary and proactive stance of the company, rather than actions resulting from external political or regulatory pressure. The politicization may manifest through product design choices, moderation strategies, public communications, or corporate affiliations, and raises concerns about bias, fairness, and the influence of corporate power on democratic discourse.

Tech companies and CEOs often associated with politicization discussions include:

- Meta (Facebook, Instagram), and CEO Mark Zuckerberg

- Google and its executives (Sundar Pichai, YouTube's leadership)

- Twitter/X and Elon Musk

- Amazon and Jeff Bezos

- Apple and Tim Cook

- Reddit

Assign:
-  Return 1 (Politicization) if a tech company or its CEO proposes or supports policies on content moderation, ideas, or movements that are commonly associated with a particular political ideology, even if not linked to a political party directly.
-  Return 1 (Politicization) in cases of a big tech company executives making a commentary on the government or governmental institutions or on ideas that belong to a cetrain side of the political spectrum .
-  Return 1 (Politicization) if the company's leadership is taking voluntary action (e.g., policy changes, advocacy, public messaging) that aligns with values typically associated with the left or right 
-  Return 0 (Not Politicization) if the post dicsusses external pressure by a political party or any other entity to a tech company.
-  Return 0 (Not Politicization) if the post discusses the politicization of a company that does not belong to the list above
-  Return 0 (Not Politicization) if the post discusses product pricing, UX, or design decisions unless they are directly tied to political content

Examples:
Post: "Facebook in 'bare-knuckle' fight with TikTok. The chief executive of a political consulting firm has responded to a report alleging Meta paid his company to "undermine" TikTok."
Label: 0
Rationale: Facebook used a firm to fight tiktok. They did not align themselves with any political party or identified with certain political ideas.

Post: “Amazon Studios Blasted for Censoring Scenes in Christmas Classic Its A Wonderful Life”
Label: 0
Rationale: The Post mentions how Amazon is censoring a movie but censorship purely is not relevant to our topic since we lack context to conclude that this particular censorship is in favor of some political party to conclude the politicization of Amazon

Post: "Israel Is Buying Google Ads to Discredit the UNs Top Gaza Aid Agency"
Label: 1
Rationale: Google Ads is allowing Israel to buy and work with their products, despite the Irsael-Palestinian conflict. While on the other hand Google has seized all operations with Russia. THis implies that google is taking a side and is keeping double standards

Post: "Crowd Outside Mark Zuckerberg's Home Protests Political Disinformation on Facebook: 'Wake the Zuck Up"
Label: 1
Rationale: The post describes a protest outside Mark Zuckerberg's home regarding political disinformation on Facebook. This implies that Facebook, under Zuckerberg's leadership, is perceived as allowing or not adequately addressing political disinformation, which can be seen as a form of voluntary politicization. 

Post: "ChatGPT Declares Trump's Physical Results 'Virtually Impossible': 'Usually Only Seen in Elite Bodybuilders"
Label: 1
Rationale: The post is mentioning how chat gpt commented on Trump's Physical Impossible. The post is a 1 since an LLM has commented negatively on a political figure.

Post: “Meta denies forcing accounts to follow Donald Trump, claims hiding Democrat hashtags is a bug | Users aren't convinced”
Label: 1
Rationale: According to the user's post META is suspected to promote Trump/Republican content over Democratic content. This clearly suggests the Politicization of META since they re actively picking a side, working with Trump and promote Republican content.

Now classify the following Reddit post:

Passage: {passage}

Label:

Rationale:
'''

def build_prompt(passage: str) -> str:
    return NEW_PROMPT.format(passage=passage)

def classify_politicization(passage: str) -> str:
    prompt = build_prompt(passage)
    try:
        signal.alarm(120)  # 2-minute timeout
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature = 0
        )
        signal.alarm(0)  # Cancel alarm

        output = response.choices[0].message.content.strip()
        label_match = re.search(r'Label:\s*(\d)', output)
        rationale_match = re.search(r'Rationale:\s*(.*)', output, re.DOTALL)

        label = label_match.group(1) if label_match else '0'
        rationale = rationale_match.group(1).strip() if rationale_match else ''

        return label, rationale
    except Exception as e:
        signal.alarm(0)
        raise e
    
def save_dataframe(df: pd.DataFrame, output_path: str):
    try:
        df.to_excel(output_path, index=False)
    except Exception as save_err:
        fallback_path = output_path.replace('.xlsx', '_fallback.csv')
        df.to_csv(fallback_path, index=False)

def process_excel(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    if 'llm_label' not in df.columns:
        df['llm_label'] = None
    if 'rationales' not in df.columns:
        df['rationales'] = None

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if pd.notnull(df.at[idx, 'llm_label']):
                continue  

            passage = f"{row['title']} {row['selftext']}"
            label, rationale = classify_politicization(passage)
            df.at[idx, 'llm_label'] = label
            df.at[idx, 'rationales'] = rationale

        save_dataframe(df, output_path)
    except Exception as e:
        print(f"Error or timeout at row {idx}: {e}")
        save_dataframe(df, output_path)
        sys.exit(1)

    print(f"Finished processing. Output saved to {output_path}")

if __name__ == "__main__":
    input_csv_file = "/Users/panayiotisfotopoulos/Thesis/tech_classification_dataset/classified_dataset_fallback.csv"
    output_xlsx_file = "/Users/panayiotisfotopoulos/Thesis/tech_classification_dataset/classified_dataset(1).xlsx"
    process_excel(input_csv_file, output_xlsx_file)
