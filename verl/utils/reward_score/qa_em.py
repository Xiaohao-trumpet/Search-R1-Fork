# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


# ---------------------------------------------------------------------------
# Reward shaping (R+): dense F1 answer reward + format gate + retrieval-utility
# bonus + over-search penalty. See IMPROVEMENT_DESIGN_zh.md sections 4.1 / 5.1.
#
# Rationale (why each term):
#  - F1 instead of binary EM: binary EM is too sparse -> most GRPO groups are
#    all-zero -> zero advantage -> no gradient (problem P1). F1 gives partial
#    credit to correct-but-reworded / near-miss answers. R-Search (2506.04185)
#    reports F1 beats EM by ~52.6% avg. (P1, P3)
#  - answer_word_cap: F1/Cover-EM are hackable by verbosity; cap the answer to
#    N words before scoring (R1-Searcher++ uses <=10). (P3)
#  - format as a GATE (not a positive bonus): positive format rewards get hacked
#    ("One Token to Fool", 2507.08794). Answer reward only counts when the
#    <think>/<search>/<information>/<answer> sequence is valid. (P3)
#  - retrieval-utility bonus: give partial credit when the gold answer appears in
#    a retrieved <information> block even if the final answer is wrong -> teaches
#    the search sub-skill and densifies signal (AutoRefine 2505.11277). (P2, P1)
#  - over-search penalty: small penalty for extra searches to curb over-searching
#    (StepSearch / R1-Searcher++). Kept small so it does not suppress needed hops.
# ---------------------------------------------------------------------------

def f1_check(prediction, golden_answers, word_cap=None):
    """Token-level F1 between prediction and the best-matching golden answer.

    If word_cap is set, the (normalized) prediction is truncated to word_cap
    tokens first -> neutralizes the verbosity/keyword-stuffing exploit.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()
    if word_cap is not None and word_cap > 0:
        pred_tokens = pred_tokens[:word_cap]
    best_f1 = 0.0
    for golden_answer in golden_answers:
        gold_tokens = normalize_answer(golden_answer).split()
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            best_f1 = max(best_f1, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def count_searches(text):
    """Number of completed <search>...</search> the model emitted (assistant part
    only, so the literal <search> in the prompt instructions is not counted)."""
    m = re.search(r"<\|im_start\|>assistant", text)
    content = text[m.end():] if m else text
    return content.count("</search>")


def compute_score_f1_shaped(solution_str, ground_truth,
                            method='strict',
                            format_score=0.,
                            answer_word_cap=10,
                            retrieval_score=0.2,
                            search_penalty=0.05,
                            free_searches=1,
                            invalid_format_score=0.0,
                            score=1.):
    """Dense, shaped reward (R+). Returns a float in roughly [0, 1].

    valid format:
        base   = F1(answer[:cap], gold)                      # dense answer reward
        if base ~ 0 and gold in retrieved info: base = retrieval_score  # AutoRefine
        base  -= search_penalty * max(0, #searches - free_searches)     # efficiency
        return clamp(base, 0, 1)
    invalid format:
        return invalid_format_score (default 0)              # format acts as a gate
    """
    # Local import keeps this file's top-level deps minimal and avoids cycles.
    from verl.utils.reward_score.qa_em_format import is_valid_sequence, is_retrieval_correct

    is_valid_format, _ = is_valid_sequence(solution_str)
    answer = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"[f1_shaped] Golden answers: {ground_truth['target']}")
        print(f"[f1_shaped] Extracted answer: {answer} | valid_format={is_valid_format}")
        print(f"[f1_shaped] Solution string: {solution_str}")

    if not is_valid_format:
        return invalid_format_score

    base = f1_check(answer, ground_truth['target'], word_cap=answer_word_cap) if answer is not None else 0.0

    if base <= 1e-8 and is_retrieval_correct(solution_str, ground_truth['target']):
        base = retrieval_score

    n_search = count_searches(solution_str)
    base = base - search_penalty * max(0, n_search - free_searches)

    return max(0.0, min(1.0, base))
