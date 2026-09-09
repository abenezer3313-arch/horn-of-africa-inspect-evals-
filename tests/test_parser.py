from scrape_evals import (
    detect_code_switching,
    detect_scripts,
    normalize_unicode,
    parse_record,
)


def test_unicode_normalization_removes_zero_width_space():
    text = "hello\u200bworld"
    result = normalize_unicode(text)

    assert "\u200b" not in result
    assert result == "helloworld"


def test_english_is_detected_as_latin():
    result = detect_scripts("hello world")

    assert "latin-script" in result


def test_amharic_is_detected_as_ethiopic():
    result = detect_scripts("ሰላም")

    assert "ethiopic-script" in result


def test_mixed_amharic_english_is_code_switched():
    result = detect_code_switching("ሰላም hello")

    assert result is True


def test_english_only_is_not_code_switched():
    result = detect_code_switching("hello world")

    assert result is False


def test_parse_record_contains_required_fields():
    result = parse_record("ሰላም hello", record_id=1)

    assert result["id"] == 1
    assert result["raw_text"] == "ሰላም hello"
    assert "normalized_text" in result
    assert "scripts" in result
    assert "code_switched" in result
    assert "tokens" in result
    assert result["code_switched"] is True
