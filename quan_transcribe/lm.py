"""Optional CTC prefix beam search with a word-level KenLM n-gram model (shallow fusion).

Units are Gemma tokenizer pieces; a piece starting with "▁" begins a new word. A word is scored by the LM when it is
completed (the next word starts, or the utterance ends):

    score = log P_ctc(prefix) + alpha * ln P_lm(words) + beta * n_words

Words the LM has never seen, and unfinished words that are not a prefix of any known word, get `unk_offset`.
Requires `pip install kenlm` (built from source; needs a C++ compiler).
"""

import math

import numpy as np

LN10 = math.log(10.0)
NEG_INF = float("-inf")
WORD_START = "▁"


def _lse(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


class _Beam:
    __slots__ = ("pb", "pnb", "lm_state", "lm_score", "n_words", "partial")

    def __init__(self, lm_state, lm_score: float, n_words: int, partial: str):
        self.pb = NEG_INF  # log prob of the prefix, ending in blank
        self.pnb = NEG_INF  # ... ending in its last unit
        self.lm_state = lm_state  # KenLM state after the completed words
        self.lm_score = lm_score  # alpha/beta-weighted LM score of the completed words
        self.n_words = n_words
        self.partial = partial  # characters of the word in progress


class KenLMDecoder:
    def __init__(
        self,
        lm_path: str,
        pieces: list[str],
        unigrams: list[str] | None = None,
        alpha: float = 0.3,
        beta: float = 1.0,
        unk_offset: float = -10.0,
        blank: int = 0,
    ):
        """pieces: text of every CTC unit (index = unit id); blank/unk entries are ignored."""
        import kenlm  # optional dependency

        self.lm = kenlm.Model(lm_path)
        self.pieces = pieces
        self.alpha, self.beta, self.unk_offset, self.blank = alpha, beta, unk_offset, blank
        self.unigrams = set(unigrams) if unigrams else None
        self.prefixes = None
        if self.unigrams:
            self.prefixes = {w[:i] for w in self.unigrams for i in range(1, len(w) + 1)}

    def _word_score(self, state, word: str):
        out = type(state)()
        s = self.lm.BaseScore(state, word, out) * LN10 * self.alpha + self.beta
        if self.unigrams is not None and word not in self.unigrams:
            s += self.unk_offset
        return out, s

    def _partial_penalty(self, partial: str) -> float:
        if not partial or self.prefixes is None or partial in self.prefixes:
            return 0.0
        return self.unk_offset

    def decode(
        self, log_probs: np.ndarray, beam_width: int = 32, token_min_logp: float = -5.0, beam_prune_logp: float = -10.0
    ) -> list[int]:
        """log_probs: (T, V) CTC log-probabilities of one utterance. Returns the best unit-id sequence."""
        import kenlm

        start = kenlm.State()
        self.lm.BeginSentenceWrite(start)
        root = _Beam(start, 0.0, 0, "")
        root.pb = 0.0
        beams: dict[tuple, _Beam] = {(): root}

        for lp in log_probs:
            cands = np.flatnonzero(lp >= token_min_logp)
            if len(cands) == 0:
                cands = np.array([int(lp.argmax())])
            new: dict[tuple, _Beam] = {}

            def child(prefix: tuple, parent: _Beam, unit: int) -> _Beam:
                key = prefix + (unit,)
                if key not in new:
                    old = beams.get(key)  # same prefix last frame: LM part is unchanged
                    if old is not None:
                        new[key] = _Beam(old.lm_state, old.lm_score, old.n_words, old.partial)
                    elif self.pieces[unit].startswith(WORD_START):
                        state, score, n = parent.lm_state, parent.lm_score, parent.n_words
                        if parent.partial:
                            state, s = self._word_score(state, parent.partial)
                            score, n = score + s, n + 1
                        new[key] = _Beam(state, score, n, self.pieces[unit][len(WORD_START) :])
                    else:
                        partial = parent.partial + self.pieces[unit]
                        new[key] = _Beam(parent.lm_state, parent.lm_score, parent.n_words, partial)
                return new[key]

            def same(prefix: tuple, beam: _Beam) -> _Beam:
                if prefix not in new:
                    new[prefix] = _Beam(beam.lm_state, beam.lm_score, beam.n_words, beam.partial)
                return new[prefix]

            for prefix, beam in beams.items():
                total = _lse(beam.pb, beam.pnb)
                for c in cands:
                    c = int(c)
                    p = float(lp[c])
                    if c == self.blank:
                        nb = same(prefix, beam)
                        nb.pb = _lse(nb.pb, total + p)
                    elif prefix and c == prefix[-1]:
                        nb = same(prefix, beam)  # repeated unit without a blank in between: collapses
                        nb.pnb = _lse(nb.pnb, beam.pnb + p)
                        ch = child(prefix, beam, c)  # repeated unit after a blank: a new unit
                        ch.pnb = _lse(ch.pnb, beam.pb + p)
                    elif self.pieces[c]:
                        ch = child(prefix, beam, c)
                        ch.pnb = _lse(ch.pnb, total + p)

            scored = sorted(
                ((_lse(b.pb, b.pnb) + b.lm_score + self._partial_penalty(b.partial), k) for k, b in new.items()),
                reverse=True,
            )
            best = scored[0][0]
            beams = {k: new[k] for s, k in scored[:beam_width] if s >= best + beam_prune_logp}

        # Finish the last word and the sentence.
        def final(item):
            prefix, b = item
            state, score = b.lm_state, b.lm_score
            if b.partial:
                state, s = self._word_score(state, b.partial)
                score += s
            end = type(state)()
            score += self.lm.BaseScore(state, "</s>", end) * LN10 * self.alpha
            return _lse(b.pb, b.pnb) + score

        return list(max(beams.items(), key=final)[0])
