from bitcliff_pipeline.suites.retrieval import generate_items, grade


def test_generation_is_deterministic():
    a = generate_items(n_items=5, n_pairs=4, seed=1301)
    b = generate_items(n_items=5, n_pairs=4, seed=1301)
    assert a == b


def test_different_seed_differs():
    a = generate_items(n_items=5, n_pairs=4, seed=1301)
    b = generate_items(n_items=5, n_pairs=4, seed=1302)
    assert a != b


def test_item_shape():
    (item,) = generate_items(n_items=1, n_pairs=4, seed=7)
    assert item.suite == "retrieval"
    assert item.id == "retrieval-7-000"
    assert len(item.expected) == 2
    for code in item.expected:
        assert code in item.prompt  # the answer is present in the context
    assert item.prompt.count(":") >= 4  # all pairs listed


def test_codes_unique_within_item():
    (item,) = generate_items(n_items=1, n_pairs=8, seed=7)
    # every code in the prompt context appears exactly once
    codes = [tok.strip(".,") for tok in item.prompt.split() if tok.strip(".,").isdigit()]
    assert len(codes) == len(set(codes))


def test_grade_correct_partial_wrong():
    (item,) = generate_items(n_items=1, n_pairs=4, seed=7)
    a, b = item.expected
    assert grade(item, f"The two codes are {a} and {b}.") == "correct"
    assert grade(item, f"I only remember {a}.") == "partial"
    assert grade(item, "No idea, maybe 0000 and 1111?" ) == "wrong"
