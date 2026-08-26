from bitcliff_pipeline.suites.spectacle import grade, load_items


def test_load_and_grade(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text(
        "prompts:\n"
        "  - {id: spec-001, prompt: 'Write a haiku about databases.'}\n"
        "  - {id: spec-002, prompt: 'Explain gravity to a pirate.'}\n"
    )
    items = load_items(p)
    assert [i.id for i in items] == ["spec-001", "spec-002"]
    assert all(i.suite == "spectacle" and i.expected is None for i in items)
    assert grade(items[0], "anything at all") == "unscored"
