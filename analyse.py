import json
from groq import Groq


def analyse_meeting(transcript_text, api_key):
    """
    Sends transcript to Llama 3.1 70B via Groq free API.
    Returns structured JSON with summary, action items, decisions, topics.
    """
    client = Groq(api_key=api_key)

    prompt = f"""
You are an expert meeting analyst. Carefully read the following meeting transcript
and return ONLY a valid JSON object. No explanation. No extra text. Just the JSON.

The JSON must follow this exact structure:
{{
  "summary": "A clear 2-3 sentence overview of what the entire meeting was about",
  "action_items": [
    {{
      "task": "specific task that needs to be done",
      "owner": "SPEAKER_A or SPEAKER_B — whoever is responsible",
      "deadline": "deadline if mentioned in the meeting, otherwise write not specified"
    }}
  ],
  "key_decisions": [
    "First important decision made in the meeting",
    "Second important decision made in the meeting"
  ],
  "topics_discussed": [
    "First main topic",
    "Second main topic",
    "Third main topic"
  ],
  "meeting_mood": "Overall tone of the meeting — positive / neutral / tense / mixed"
}}

MEETING TRANSCRIPT:
{transcript_text}

Remember: Return ONLY the JSON object. No markdown. No code blocks. No extra words.
"""

    print("Sending transcript to Llama 3.1 70B via Groq...")

    response = client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )

    raw = response.choices[0].message.content.strip()

    # safely parse the JSON response
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # if LLM added extra text, find just the JSON part
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            result = json.loads(raw[start:end])
        else:
            print("⚠️ Could not parse LLM response as JSON. Raw response:")
            print(raw)
            result = {
                "summary": raw,
                "action_items": [],
                "key_decisions": [],
                "topics_discussed": [],
                "meeting_mood": "unknown"
            }

    return result


if __name__ == "__main__":
    # test with a sample transcript
    sample_transcript = """
    SPEAKER_A: Good morning everyone. Today we need to finalize the product launch date.
    SPEAKER_B: I think we should aim for March 15th. The development team needs two more weeks.
    SPEAKER_A: That works. Can you prepare the marketing materials by March 10th?
    SPEAKER_B: Yes, I will handle that. I also need the final logo from the design team.
    SPEAKER_A: I will follow up with the design team today and get it to you by end of week.
    SPEAKER_B: Perfect. So our decision is March 15th launch with a review on March 12th.
    SPEAKER_A: Agreed. Let us also set up a customer feedback form before launch.
    SPEAKER_B: I will create that as well. Should be ready by March 8th.
    """

    import os
    api_key = os.environ.get("GROQ_API_KEY", "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")

    result = analyse_meeting(sample_transcript, api_key)

    print("\n--- MEETING ANALYSIS ---")
    print(f"\nSUMMARY:\n{result['summary']}")

    print(f"\nACTION ITEMS:")
    for item in result["action_items"]:
        print(f"  ✅ {item['task']}")
        print(f"     Owner: {item['owner']} | Deadline: {item['deadline']}")

    print(f"\nKEY DECISIONS:")
    for d in result["key_decisions"]:
        print(f"  🔹 {d}")

    print(f"\nTOPICS DISCUSSED:")
    for t in result["topics_discussed"]:
        print(f"  📌 {t}")

    print(f"\nMEETING MOOD: {result['meeting_mood']}")
