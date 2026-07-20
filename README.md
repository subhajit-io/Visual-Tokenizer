# Visual Tokenizer

A visual, interactive tool to understand how Byte Pair Encoding (BPE) works. This application allows users to train a BPE tokenizer on custom text and observe the tokenization process in real-time.

## Live Demo
You can see the project in action here: [https://visual-tokenizer-gilt.vercel.app/](https://visual-tokenizer-gilt.vercel.app/)

## Acknowledgments & References
This project was inspired by and built using knowledge from the following resources:
*   **Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out." video:** The conceptual foundation for the BPE algorithm and tokenization process.
*   **OpenAI's `gpt4_tokenizer` repository:** Served as a primary technical reference for implementation patterns and regex-based tokenization logic.

## How it Works
This tool is a window into how Large Language Models (LLMs) like GPT-4 "read" text. Instead of seeing words as humans do, they process text as numerical identifiers created through **Byte Pair Encoding (BPE)**.

### 1. The Core Logic: Merges
The `merge_ids` function is the engine of the BPE algorithm. It works by scanning the text for the most frequently occurring pair of adjacent bytes (e.g., if "t" and "h" appear together often, it merges them into a single new unit: "th").
*   **The Loop**: The tokenizer iteratively finds the most frequent pair in the training text, merges them into a new, unique ID, and updates the vocabulary.
This allows the model to compress common sequences of characters into single "tokens," making the processing of text much more efficient than looking at individual characters.

### 2. The Sliders: Controlling the "Brain"
The sliders in your application allow you to manually adjust the complexity and capacity of your tokenizer:

*   **Train Merges Slider**:
    *   This controls **how many training iterations** the model performs.
    *   Higher values allow the tokenizer to learn longer, more complex sequences (like whole words or common word fragments).
    *   Low values keep the tokenizer focused on simple character combinations.
*   **Test Merges Slider**:
    *   This controls the **limit of tokens** allowed during the testing/inference phase.
    *   It determines how many of the "learned rules" from the training phase are actually applied to your test text.
    *   If this is lower than the training value, the tokenizer will effectively "ignore" some of the complex merges it learned, resulting in more granular (smaller) tokens.

### 3. How the Workflow Functions
1.  **Input**: You provide text in the **Training** section, which the app converts into raw bytes.
2.  **Learning**: The app runs the BPE algorithm to create a table of "merge rules."
3.  **Visualization**: The UI displays these merges in real-time, showing you exactly how the text is transformed from bytes into tokens.
4.  **Inference**: In the **Test** section, the app applies those learned rules to your new text to show you how a GPT-style model would break that specific input down into tokens.
