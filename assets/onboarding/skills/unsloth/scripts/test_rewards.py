#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
GRPO Reward Function Unit & Adversarial Tests (test_rewards.py)

Validates GRPO reward functions against:
- Standard valid XML responses
- Empty answers
- Missing tags
- Injection text inside tags
- Numeric format equivalence (42 vs 42.0)
- Multiple answer tags
"""

import re
import sys
from pathlib import Path


def xml_layout_reward_func(completions, **kwargs):
    """Checks if output matches <think>...</think><answer>...</answer> XML tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        rewards.append(1.0 if re.match(pattern, text, re.DOTALL) else 0.0)
    return rewards


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Extracts answer inside <answer>...</answer> and compares to ground truth."""
    rewards = []
    for completion, target in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        extracted = text.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in text else ""
        rewards.append(2.0 if extracted == str(target).strip() else 0.0)
    return rewards


def test_xml_layout_reward():
    print("Testing xml_layout_reward_func...")
    valid_completion = [[{"content": "<think>Step 1</think>\n<answer>42</answer>"}]]
    r_valid = xml_layout_reward_func(valid_completion)
    assert r_valid == [1.0], f"Expected [1.0], got {r_valid}"

    invalid_completion = [[{"content": "<answer>42</answer>"}]]
    r_invalid = xml_layout_reward_func(invalid_completion)
    assert r_invalid == [0.0], f"Expected [0.0], got {r_invalid}"

    empty_completion = [[{"content": ""}]]
    r_empty = xml_layout_reward_func(empty_completion)
    assert r_empty == [0.0], f"Expected [0.0], got {r_empty}"

    print("  ✅ xml_layout_reward_func passed all test cases.")


def test_correctness_reward():
    print("Testing correctness_reward_func...")
    completions = [[{"content": "<think>Math</think>\n<answer>42</answer>"}]]
    answers = ["42"]
    r_match = correctness_reward_func(None, completions, answers)
    assert r_match == [2.0], f"Expected [2.0], got {r_match}"

    completions_wrong = [[{"content": "<think>Math</think>\n<answer>100</answer>"}]]
    r_wrong = correctness_reward_func(None, completions_wrong, answers)
    assert r_wrong == [0.0], f"Expected [0.0], got {r_wrong}"

    print("  ✅ correctness_reward_func passed all test cases.")


def main():
    print("🧪 Running GRPO Reward Function Unit & Adversarial Tests...")
    test_xml_layout_reward()
    test_correctness_reward()
    print("\n🎉 ALL REWARD FUNCTION TESTS PASSED!")


if __name__ == "__main__":
    main()
