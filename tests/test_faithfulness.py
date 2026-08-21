from explainrec.eval.faithfulness import check_faithfulness

REPORT = {
    "objective": {"base": 4213.5, "modified": 4100.2, "delta_pct": -2.69},
    "cold_item_exposure": {"base_total": 3775.0, "modified_total": 0.0},
    "per_user_slate_rating_distribution": {
        "base": {"min": 1.2, "median": 3.4, "max": 4.9},
    },
}


def test_grounded_numbers_all_match():
    text = (
        "The objective dropped from 4213.5 to 4100.2, a 2.69% decrease. "
        "Cold item exposure fell from 3775 to 0."
    )
    result = check_faithfulness(REPORT, text)
    assert result.unmatched == []
    assert result.match_rate == 1.0


def test_fabricated_number_is_flagged():
    text = "Cold item exposure fell to 500 users, well below the target."
    result = check_faithfulness(REPORT, text)
    assert 500.0 in result.unmatched


def test_rounding_within_tolerance_matches():
    text = "About 3776 cold-item exposures at baseline, roughly 3% higher objective."
    result = check_faithfulness(REPORT, text, tolerance=1.0)
    assert 3776.0 in result.matched


def test_small_integers_are_not_scored():
    text = "There are 2 users whose slates barely changed, 1 constraint was removed."
    result = check_faithfulness(REPORT, text)
    assert result.claimed == []
    assert result.match_rate == 1.0


def test_percent_delta_recognized():
    text = "Efficiency dropped by 2.69 percent."
    result = check_faithfulness(REPORT, text)
    assert result.unmatched == []


def test_empty_explanation_is_fully_faithful():
    result = check_faithfulness(REPORT, "No numeric claims here at all.")
    assert result.claimed == []
    assert result.match_rate == 1.0


def test_thousands_separator_is_not_split_into_two_numbers():
    report = {"objective": {"base": 38481.1, "modified": 42367.3}}
    text = "The objective rose from 38,481.1 to 42,367.3."
    result = check_faithfulness(report, text)
    assert result.unmatched == []
    assert 38.0 not in result.claimed
    assert 481.0 not in result.claimed


def test_fraction_restated_as_percent_matches():
    # "top 10%" here names a report field (top_10pct_...), not a value, so
    # the check is on whether the value 1.0 is faithfully restated as "100%".
    report = {"item_exposure_concentration": {"exposure_share": {"modified": 1.0}}}
    text = "The most exposed items would absorb 100% of all exposure."
    result = check_faithfulness(report, text)
    assert result.unmatched == []


def test_unmatched_number_reports_its_source_sentence():
    text = (
        "The objective dropped from 4213.5 to 4100.2. "
        "Cold item exposure fell to 500 users, well below the target."
    )
    result = check_faithfulness(REPORT, text)
    assert result.unmatched == [500.0]
    assert result.unmatched_sentences == [
        "Cold item exposure fell to 500 users, well below the target."
    ]


def test_multiple_unmatched_numbers_map_to_their_own_sentences():
    text = (
        "Provider payouts rose to 900 dollars. "
        "Meanwhile churn hit 750 users this quarter."
    )
    result = check_faithfulness(REPORT, text)
    assert result.unmatched == [900.0, 750.0]
    assert result.unmatched_sentences == [
        "Provider payouts rose to 900 dollars.",
        "Meanwhile churn hit 750 users this quarter.",
    ]


def test_single_sentence_with_no_terminal_punctuation():
    text = "Exposure jumped to 999 users"
    result = check_faithfulness(REPORT, text)
    assert result.unmatched == [999.0]
    assert result.unmatched_sentences == ["Exposure jumped to 999 users"]
