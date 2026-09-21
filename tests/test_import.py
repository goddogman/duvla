def test_import() -> None:
    import duvla

    assert duvla.__name__ == "duvla"
    assert duvla.__model_name__ == "Duvla"
