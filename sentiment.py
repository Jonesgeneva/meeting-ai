from textblob import TextBlob


def analyse_sentiment(labelled_lines):
    """
    Scores each speaker's sentiment using TextBlob.
    Polarity: -1.0 = very negative, 0 = neutral, +1.0 = very positive
    Returns per-speaker average score and label.
    """
    speaker_scores = {}
    speaker_lines_count = {}

    for item in labelled_lines:
        speaker = item["speaker"]
        text = item["text"]

        # skip very short lines — not enough text for reliable sentiment
        if len(text.split()) < 3:
            continue

        score = TextBlob(text).sentiment.polarity

        if speaker not in speaker_scores:
            speaker_scores[speaker] = []
            speaker_lines_count[speaker] = 0

        speaker_scores[speaker].append(score)
        speaker_lines_count[speaker] += 1

    results = {}

    for speaker, scores in speaker_scores.items():
        avg = round(sum(scores) / len(scores), 2)

        if avg > 0.15:
            label = "Positive"
            emoji = "😊"
        elif avg < -0.15:
            label = "Negative"
            emoji = "😟"
        else:
            label = "Neutral"
            emoji = "😐"

        results[speaker] = {
            "score": avg,
            "label": label,
            "emoji": emoji,
            "lines_analysed": speaker_lines_count[speaker]
        }

        print(f"{speaker}: {emoji} {label} (score: {avg}, lines analysed: {speaker_lines_count[speaker]})")

    return results


def get_overall_sentiment(sentiment_results):
    """
    Combines all speaker scores into one overall meeting sentiment.
    """
    all_scores = [v["score"] for v in sentiment_results.values()]
    if not all_scores:
        return {"score": 0, "label": "Neutral", "emoji": "😐"}

    avg = round(sum(all_scores) / len(all_scores), 2)

    if avg > 0.15:
        return {"score": avg, "label": "Positive", "emoji": "😊"}
    elif avg < -0.15:
        return {"score": avg, "label": "Negative", "emoji": "😟"}
    else:
        return {"score": avg, "label": "Neutral", "emoji": "😐"}


if __name__ == "__main__":
    # test with sample lines
    sample_lines = [
        {"speaker": "SPEAKER_A", "text": "This is a fantastic idea and I am really excited about moving forward."},
        {"speaker": "SPEAKER_A", "text": "The results have been great so far and the team is doing well."},
        {"speaker": "SPEAKER_B", "text": "I am not confident this will work. There are too many risks involved."},
        {"speaker": "SPEAKER_B", "text": "The timeline is too tight and I am worried we will miss the deadline."},
        {"speaker": "SPEAKER_A", "text": "We can handle it. Let us just stay focused and keep communicating."},
    ]

    print("--- SENTIMENT ANALYSIS ---")
    results = analyse_sentiment(sample_lines)

    overall = get_overall_sentiment(results)
    print(f"\nOVERALL MEETING SENTIMENT: {overall['emoji']} {overall['label']} (score: {overall['score']})")
