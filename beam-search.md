# Beam Search

**Beam search** is a heuristic search algorithm used for **decoding sequences** in tasks like machine translation, text generation, and speech recognition. It's a compromise between **greedy decoding** and **exhaustive search**, aiming to find the most likely output sequence without evaluating every possible one.

* In sequence generation, models output probabilities over the vocabulary at each time step. The naive way (greedy search) picks the most likely next word at each step — but this can lead to poor overall sequences. Exhaustive search (finding the truly most likely full sequence) is too expensive. Beam search balances between these extremes.


* Instead of keeping just **1 best sequence** (as in greedy), beam search keeps the **top *k* sequences at each step**, where *k* is the **beam width**.

    At each decoding time step:

    1. For each of the current *k* sequences:
    * Compute the probability of all possible next tokens.
    * Extend the sequence with each token.
    2. Among all the extended sequences (k × vocab size), keep the **top *k*** most probable ones.
    3. Repeat until end-of-sequence tokens are generated or a max length is reached.


* Log Probabilities

    We usually compute:

    $$
    \log P(y_1, y_2, \dots, y_T) = \sum_{t=1}^T \log P(y_t \mid y_{<t})
    $$

    This avoids numerical underflow and makes addition possible (instead of multiplying tiny probabilities).

* Example (Beam Width = 2)

    Suppose our model vocabulary is: `["I", "like", "dogs", "cats", "<eos>"]`

    1. Step 1:

    * Initial token: `<s>`
    * Model predicts:

        * "I": 0.5
        * "You": 0.3
        * "They": 0.2
    * Beam keeps top 2: `["I"], ["You"]`

    2. Step 2:

    * Expand "I":

        * "like": 0.6
        * "hate": 0.4
    * Expand "You":

        * "like": 0.7
        * "see": 0.3
    * Beam keeps top 2: `["I", "like"], ["You", "like"]`

    3. Continue expanding until `<eos>` is hit.


* Pros
    * Higher quality sequences than greedy search.
    * Computationally tractable.

* Cons:
    * Still approximate, may miss the global optimum.
    * Larger beams = slower.
    * Bias toward shorter sequences unless length normalization is used.

* Variants
    * **Length Normalization**
        * Penalizes short sequences.
        * adjusts the total log-probability of a sequence by its length (e.g., dividing by length or applying a length penalty) to prevent the beam search from favoring shorter sequences, which naturally accumulate higher (less negative) log-probabilities.
    * **Diverse Beam Search**
        * Encourages diverse sequences.
        * partitions the beam into groups and penalizes tokens that appear frequently across groups, promoting more varied candidate sequences.
    * **Stochastic Beam Search**
        * Samples among top sequences instead of deterministic choice.  
        * introduces randomness by sampling from the top-*k* candidates (weighted by probability) rather than always choosing the highest scoring ones, allowing exploration of more diverse or potentially better sequences.


* PyTorch

    ```python
    # Example vocab
    vocab = ["<s>", "I", "like", "dogs", "cats", "<eos>"]
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    # Simulated model: takes input tokens and returns logits over vocab
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids):
            # input_ids: [batch_size, seq_len]
            # We'll mock fixed logits regardless of input
            logits = torch.tensor([
                [0.0, 2.0, 1.0, 0.5, 0.3, -float("inf")],   # Step 1
                [-float("inf"), 0.1, 2.0, 1.5, 0.2, 0.5],    # Step 2
                [-float("inf"), -float("inf"), 0.5, 0.2, 0.1, 2.0],  # Step 3 (forces <eos>)
            ])
            step = input_ids.shape[1] - 1
            return logits[step].unsqueeze(0)  # [1, vocab_size]

    # Beam search decoder
    def beam_search(model, start_token, beam_width=2, max_len=5):
        model.eval()
        sequences = [([token_to_id[start_token]], 0.0)]  # list of (token_ids, log_prob)

        for step in range(max_len):
            all_candidates = []

            for seq, score in sequences:
                input_tensor = torch.tensor([seq])
                logits = model(input_tensor)  # [1, vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]

                # Expand current sequence with all possible tokens
                for idx in range(len(vocab)):
                    new_seq = seq + [idx]
                    new_score = score + log_probs[0, idx].item()
                    all_candidates.append((new_seq, new_score))

            # Select top k sequences
            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

            # Stop if all sequences end with <eos>
            if all(seq[-1] == token_to_id["<eos>"] for seq, _ in sequences):
                break

        # Decode tokens to strings
        final_seqs = [[id_to_token[idx] for idx in seq] for seq, _ in sequences]
        return final_seqs

    # Run it
    model = DummyModel()
    results = beam_search(model, start_token="<s>", beam_width=2)
    for i, seq in enumerate(results):
        print(f"Beam {i + 1}: {' '.join(seq)}")
    ```

* NumPy

    ```python
    vocab = ["<s>", "I", "like", "dogs", "cats", "<eos>"]
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    # Simulated logits at each time step
    # Rows: [vocab_size], shape = [time, vocab_size]
    logits_sequence = np.array([
        [0.0, 2.0, 1.0, 0.5, 0.3, -np.inf],       # Step 1
        [-np.inf, 0.1, 2.0, 1.5, 0.2, 0.5],       # Step 2
        [-np.inf, -np.inf, 0.5, 0.2, 0.1, 2.0],   # Step 3
    ])

    def softmax(x):
        e_x = np.exp(x - np.max(x))  # for stability
        return e_x / e_x.sum()

    def beam_search_np(logits_seq, start_token="<s>", beam_width=2, max_len=5):
        sequences = [([token_to_id[start_token]], 0.0)]  # (sequence, log_prob)

        for step in range(min(max_len, logits_seq.shape[0])):
            probs = softmax(logits_seq[step])  # Convert logits to probs
            log_probs = np.log(probs + 1e-12)  # Avoid log(0)

            all_candidates = []

            for seq, score in sequences:
                for idx in range(len(vocab)):
                    new_seq = seq + [idx]
                    new_score = score + log_probs[idx]
                    all_candidates.append((new_seq, new_score))

            # Keep top k
            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

            # Early stop
            if all(seq[-1] == token_to_id["<eos>"] for seq, _ in sequences):
                break

        decoded = [[id_to_token[i] for i in seq] for seq, _ in sequences]
        return decoded

    # Run
    results = beam_search_np(logits_sequence)
    for i, seq in enumerate(results):
        print(f"Beam {i + 1}: {' '.join(seq)}")
    ```