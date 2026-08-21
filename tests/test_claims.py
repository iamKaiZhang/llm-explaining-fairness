from explainrec.eval.claims import check_attribution, check_claims, extract_claims

REPORT = {
    "objective": {"base": 4213.5, "modified": 4100.2, "delta_pct": -2.69},
    "mean_predicted_rating": {"base": 4.468, "modified": 4.348},
    "cold_item_exposure": {"base_total": 3775.0, "modified_total": 0.0},
    "slate_size": {"base": 10, "modified": 10},
    "item_exposure_concentration": {
        "gini": {"base": 0.82, "modified": 0.88},
        "top_10pct_items_exposure_share": {"base": 0.61, "modified": 0.66},
    },
}


def test_correct_down_claim_is_verified():
    text = "The total predicted rating fell from 4213.5 to 4100.2."
    result = check_claims(REPORT, text)
    assert len(result.verified) == 1
    assert result.verified[0].metric == "objective"
    assert result.verified_rate == 1.0


def test_wrong_direction_is_contradicted():
    # 3775 appears in the report, so the number check alone would pass this;
    # the direction is what gives the fabrication away.
    text = "Cold item exposure rose to 3775."
    result = check_claims(REPORT, text)
    assert len(result.contradicted) == 1
    assert result.contradicted[0].metric == "cold_item_exposure"
    assert result.verified_rate == 0.0


def test_flat_claim_verified_when_metric_unchanged():
    text = "The slate size stayed the same throughout."
    result = check_claims(REPORT, text)
    assert len(result.verified) == 1
    assert result.verified[0].direction == "flat"


def test_negated_direction_is_flipped():
    # objective actually went down, so "did not increase" is a true claim
    text = "The objective did not increase after the change."
    result = check_claims(REPORT, text)
    assert len(result.verified) == 1
    assert result.verified[0].negated


def test_metric_absent_from_report_is_unverifiable():
    text = "The exploration burden fell noticeably."
    result = check_claims(REPORT, text)
    assert len(result.unverifiable) == 1
    # unverifiable claims are excluded from the score, not counted against it
    assert result.verified_rate == 1.0


def test_mixed_claims_give_graded_score():
    text = (
        "The objective fell by about 2.7%. "
        "Exposure concentration also fell."  # gini actually rose: contradicted
    )
    result = check_claims(REPORT, text)
    assert len(result.verified) == 1
    assert len(result.contradicted) == 1
    assert result.verified_rate == 0.5


def test_metric_mention_without_direction_is_not_a_claim():
    text = "The report includes the objective and the cold item exposure."
    assert extract_claims(text) == []


def test_no_directional_claims_scores_one():
    result = check_claims(REPORT, "Nothing quantitative to see here.")
    assert result.claims == []
    assert result.verified_rate == 1.0


def test_generic_gini_not_double_counted_after_specific_metric():
    # "exposure concentration gini rose" is one claim about the exposure
    # gini, not separate claims for "exposure concentration" and "gini"
    text = "The exposure concentration gini rose from 0.82 to 0.88."
    result = check_claims(REPORT, text)
    assert len(result.claims) == 1
    assert result.claims[0].verdict == "verified"


def test_two_metrics_one_sentence_each_get_a_claim():
    text = "The objective fell while the top 10% share rose."
    result = check_claims(REPORT, text)
    metrics = {c.metric for c in result.claims}
    assert metrics == {"objective", "top_share"}
    assert all(c.verdict == "verified" for c in result.claims)


# --- coverage --------------------------------------------------------------


def test_coverage_full_when_all_direction_sentences_yield_claims():
    result = check_claims(REPORT, "The objective fell noticeably.")
    assert result.n_direction_sentences == 1
    assert result.n_claim_sentences == 1
    assert result.coverage == 1.0


def test_coverage_drops_for_unrecognized_movement_language():
    # second sentence has direction language but no lexicon metric, so it
    # is unchecked -- coverage must expose that, not hide it
    text = "The objective fell by 2.7%. Niche visibility also dropped sharply."
    result = check_claims(REPORT, text)
    assert result.n_direction_sentences == 2
    assert result.n_claim_sentences == 1
    assert result.coverage == 0.5


def test_coverage_is_one_when_no_direction_language():
    result = check_claims(REPORT, "A purely descriptive sentence.")
    assert result.coverage == 1.0


# --- field attribution -----------------------------------------------------


def test_number_in_right_field_is_attributed():
    result = check_attribution(REPORT, "The objective fell to 4100.2.")
    assert len(result.attributed) == 1
    assert result.attribution_rate == 1.0


def test_real_number_in_wrong_field_is_flagged():
    # 3775 exists in the report -- as cold item exposure, not the objective;
    # the presence check alone would pass this
    result = check_attribution(REPORT, "The objective is 3775.")
    assert len(result.wrong_field) == 1
    assert result.wrong_field[0].metric == "objective"
    assert result.attribution_rate == 0.0


def test_number_not_in_report_at_all():
    result = check_attribution(REPORT, "Cold item exposure reached 9999.")
    assert len(result.not_in_report) == 1


def test_number_without_metric_phrase_is_not_scored():
    # presence check (check_faithfulness) covers bare numbers; attribution
    # only scores numbers stated next to a metric
    result = check_attribution(REPORT, "It changed by 4100.2 overall.")
    assert result.attributions == []
    assert result.attribution_rate == 1.0


def test_identifier_numbers_are_not_scored():
    result = check_attribution(REPORT, "User 42 saw the objective fall to 4100.2.")
    assert len(result.attributions) == 1
    assert result.attributions[0].number == 4100.2


def test_digits_inside_metric_phrase_are_not_scored():
    # the "10" in "top 10%" is part of the phrase, not a numeric claim
    result = check_attribution(REPORT, "The top 10% share rose to 0.66.")
    assert len(result.attributions) == 1
    assert result.attributions[0].number == 0.66
    assert result.attributions[0].verdict == "attributed"


def test_percent_rescaling_matches_in_field():
    # 0.66 stated as 66%
    result = check_attribution(REPORT, "The top 10% share reached 66% of exposure.")
    assert len(result.attributed) == 1
