import os
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="")

def build_prompt(passage: str) -> str:
    return f"""You are a sentiment analyst tasked with labeling social media posts that have already been determined to be relevant to the perceived politicization of technology companies. 
    Your goal is to assign one of the following sentiment labels to each post:    
    •   Neutral – The post reports or describes an event without suggesting any emotional or evaluative tone. These are objective, fact-based mentions. Use this label if the tone is descriptive, explanatory, or if no clear attitude or reaction is expressed.    
    •   Negative – Use this label only when it is clear that the author of the post is being critical of a tech company or CEO for being political — or when the post reports on others expressing criticism, backlash, distrust, or negative interpretations. There must be an explicit or strongly implied negative stance toward the political behavior or perceived bias of a tech company or CEO. Avoid assigning this label based solely on topic — instead, look for evidence of judgment, blame, disapproval, or conflict.    
    •   Positive – The post implies a supportive, forgiving, or apologetic stance toward a tech company’s or CEO’s actions, particularly in response to backlash. This includes defending decisions, framing them as fair, or commending corrective actions.Focus especially on:    
        •   Whether the tone is evaluative vs. factual.    
        •   Whether criticism is directly expressed by the author or described as coming from others.    
        •   Avoid labeling as Negative just because the post mentions controversial topics — the framing and intent must be clear. Read the post first while trying to interpret it as having a positive sentiment, then again as having a negative sentiment. If both interpretations feel uncertain, ambiguous, or open to multiple readings, label the post as Neutral. Always keep in mind that this is not general sentiment classification, but sentiment in the context of the politicization of tech companies.
        
    Now classify the following Reddit post:    
    Passage: {passage}    
    Sentiment Label: [Neutral / Negative / Positive]    
    Rationale:
"""

def classify_sentiment(passage: str) -> tuple:
    prompt = build_prompt(passage)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = response.choices[0].message.content.strip()

        label_match = re.search(r"Sentiment Label:\s*(Neutral|Negative|Positive)", output, re.IGNORECASE)
        sentiment = label_match.group(1).capitalize() if label_match else "error"

        rationale_lines = [line for line in output.splitlines() if not line.lower().startswith("sentiment label")]
        rationale = "\n".join(rationale_lines).replace("Rationale:", "").strip()

        return sentiment, rationale
    except Exception as e:
        return "error", f"API Error: {str(e)}"

def process_excel(input_path: str, output_path: str):
    df = pd.read_excel(input_path)

    sentiments = []
    rationales = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        passage = f"{row['title']} {row['selftext'] if pd.notna(row['selftext']) else ''}"
        sentiment, rationale = classify_sentiment(passage)
        sentiments.append(sentiment)
        rationales.append(rationale)

    df["llm_sentiment"] = sentiments
    df["llm_sentiment_rationale"] = rationales

    df.to_excel(output_path, index=False)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    input_xlsx_file = ""
    output_xlsx_file = ""
    process_excel(input_xlsx_file, output_xlsx_file)
