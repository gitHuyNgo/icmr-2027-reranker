from icmr2027.models.vlm import build_prompt


def test_baseline_prompt():
    assert build_prompt("What color?") == (
        "Answer the question based on the image.\n\nQuestion: What color?\n\nGive a concise answer."
    )


def test_optional_context_prompt():
    prompt = build_prompt("What color?", context="The flag is red.")
    assert "Evidence:\nThe flag is red." in prompt
    assert "Question:\nWhat color?" in prompt
    assert build_prompt("Q", context="").startswith("Use the image")
