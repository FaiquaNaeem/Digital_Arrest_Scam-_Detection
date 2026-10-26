
import re
import time
import pandas as pd
import kss
INPUT_FILE = "korccvi_translated_progress.csv"     # adjust path if it's elsewhere
OUTPUT_FILE = "korccvi_pseudo_turns.csv"
MAX_SENTENCES_PER_TURN = 2                          # tunable knob


def segment_into_pseudo_turns(text: str, max_sentences_per_turn: int = 2):
    """
    Splits Korean text into pseudo-turns by grouping sentences.
    No real speaker attribution here, just chunking a monologue into
    turn-sized pieces (speaker is labeled 'unknown').
    """
    sentences = kss.split_sentences(str(text))
    turns, current = [], []
    for sent in sentences:
        current.append(sent)
        if len(current) >= max_sentences_per_turn:
            turns.append(" ".join(current))
            current = []
    if current:
        turns.append(" ".join(current))
    return turns if turns else [str(text)]


def segment_english(text: str, max_sentences_per_turn: int = 2):
    """
    Same grouping logic applied to the English translation, so the
    turn boundaries roughly line up for side-by-side reading. This is
    NOT sentence-aligned with the Korean split (different sentence
    counts are possible), it's an approximate reading aid only.
    """
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    turns, current = [], []
    for sent in sentences:
        current.append(sent)
        if len(current) >= max_sentences_per_turn:
            turns.append(" ".join(current))
            current = []
    if current:
        turns.append(" ".join(current))
    return turns if turns else [str(text)]


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows")

    records = []
    failed_rows = []
    start_time = time.time()

    for idx, row in df.iterrows():
        try:
            ko_turns = segment_into_pseudo_turns(row['transcript_clean'], MAX_SENTENCES_PER_TURN)

            # English turns, if the translation column exists
            en_turns = None
            if 'transcript_en' in df.columns and pd.notna(row.get('transcript_en')):
                en_turns = segment_english(row['transcript_en'], MAX_SENTENCES_PER_TURN)

            for turn_id, ko_text in enumerate(ko_turns):
                en_text = en_turns[turn_id] if en_turns and turn_id < len(en_turns) else None
                records.append({
                    "conversation_id": row['id'],
                    "label": row['label'],
                    "turn_id": turn_id,
                    "speaker": "unknown",
                    "text": ko_text,
                    "text_en": en_text
                })

        except Exception as e:
            failed_rows.append((row['id'], str(e)))

        # progress logging every 100 rows
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(df) - (idx + 1)) / rate if rate > 0 else 0
            print(f"[{idx + 1}/{len(df)}] elapsed={elapsed:.1f}s rate={rate:.1f}rows/s eta={remaining:.1f}s")

    turns_df = pd.DataFrame(records)
    turns_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f} seconds")
    print(f"Saved to {OUTPUT_FILE}")

    # ---- Verification report ----
    print("\n" + "=" * 50)
    print("VERIFICATION REPORT")
    print("=" * 50)
    print(f"Input transcripts: {len(df)}")
    print(f"Output pseudo-turns: {len(turns_df)}")
    print(f"Failed rows: {len(failed_rows)}")
    if failed_rows:
        print("Failed row ids:", [r[0] for r in failed_rows[:20]])

    turns_per_conv = turns_df.groupby('conversation_id').size()
    print(f"\nTurns per conversation: avg={turns_per_conv.mean():.1f}, "
          f"min={turns_per_conv.min()}, max={turns_per_conv.max()}")

    single_turn = (turns_per_conv == 1).sum()
    print(f"Conversations with only 1 turn: {single_turn} "
          f"({100 * single_turn / len(turns_per_conv):.1f}%)")

    # ---- Sample check: 10 random conversations, length vs turn count ----
    print("\n--- Random sample: original length vs turns produced ---")
    sample_ids = df['id'].sample(min(10, len(df)), random_state=42)
    for conv_id in sample_ids:
        orig_len = len(str(df.loc[df['id'] == conv_id, 'transcript_clean'].iloc[0]))
        n_turns = turns_per_conv.get(conv_id, 0)
        print(f"  id={conv_id}: {orig_len} chars -> {n_turns} turns")

    # ---- 3 full example conversations across labels ----
    print("\n" + "=" * 50)
    print("EXAMPLE CONVERSATIONS")
    print("=" * 50)
    example_ids = []
    for label_val in df['label'].unique():
        candidate = df[df['label'] == label_val]['id'].iloc[0]
        example_ids.append(candidate)
    if len(example_ids) < 3 and len(df) > 2:
        example_ids.append(df['id'].iloc[len(df) // 2])

    for conv_id in example_ids[:3]:
        sub = turns_df[turns_df['conversation_id'] == conv_id]
        label_val = df.loc[df['id'] == conv_id, 'label'].iloc[0]
        print(f"\n--- Conversation {conv_id} (label={label_val}) ---")
        for _, r in sub.iterrows():
            print(f"  Turn {r['turn_id']} [KO]: {r['text'][:150]}")
            if pd.notna(r['text_en']):
                print(f"  Turn {r['turn_id']} [EN]: {r['text_en'][:150]}")


if __name__ == "__main__":
    main()
