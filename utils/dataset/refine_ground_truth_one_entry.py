import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

LABELS_PATH = BASE_DIR / "QVED-CLEANED" / "fine_grained_labels.json"

llm = AzureChatOpenAI(
    model_name="gpt-4o",
    temperature=0.2,
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

PROMPT_TEMPLATE = """You are a fitness coach giving instant verbal feedback on someone's "{exercise}" exercise.

Existing feedbacks for this exercise:
{existing_feedbacks}

Detected issues:
{labels_descriptive}

Reference coach comments:
{coach}

If one of the EXISTING FEEDBACKS above perfectly fits the detected issues, output EXACTLY that feedback (word-for-word).
Otherwise, create a NEW feedback following these rules:

RULES:
1. Output exactly 1 sentence, maximum 10 words.
2. Say only the most critical issue — nothing else.
3. Direct coach tone, spoken out loud, no fluff.
4. Do not repeat the exercise name.
5. Output ONLY the sentence, no punctuation at the end."""


def refine_entry(entry, existing_feedbacks=None):
    exercise = entry["exercise"]
    labels_descriptive = entry.get("labels_descriptive", [])
    coach = entry.get("coach", [])

    if existing_feedbacks is None:
        existing_feedbacks = []

    labels_text = "\n".join(f"- {l}" for l in labels_descriptive) if labels_descriptive else "- none"
    coach_text = "\n".join(f"- {c}" for c in coach) if coach else "- none"
    
    if existing_feedbacks:
        existing_text = "\n".join(f"{i+1}. {fb}" for i, fb in enumerate(existing_feedbacks))
    else:
        existing_text = "(none yet)"

    prompt = PROMPT_TEMPLATE.format(
        exercise=exercise,
        existing_feedbacks=existing_text,
        labels_descriptive=labels_text,
        coach=coach_text,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def main():
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} entries from {LABELS_PATH}\n")

    entry = random.choice(labels)
    print(f"Video:             {entry['video_path']}")
    print(f"Exercise:          {entry['exercise']}")
    print(f"Labels descriptive:")
    for l in entry.get("labels_descriptive", []):
        print(f"  - {l}")
    print(f"Coach comments:")
    for c in entry.get("coach", []):
        print(f"  - {c}")

    print(f"\nRefining with LLM...")
    refined = refine_entry(entry)

    print(f"\nRefined feedback:")
    print(f"  {refined}")


if __name__ == "__main__":
    main()
