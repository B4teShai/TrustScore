alter table user_feedback add column if not exists issue_category text;
alter table user_feedback add column if not exists corrected_component text;
alter table user_feedback add column if not exists expected_risk_level text;

do $$
begin
    if not exists (
        select 1 from pg_constraint where conname = 'chk_user_feedback_issue_category'
    ) then
        alter table user_feedback
            add constraint chk_user_feedback_issue_category
            check (
                issue_category is null or issue_category in (
                    'score_too_high',
                    'score_too_low',
                    'wrong_product',
                    'wrong_seller',
                    'wrong_reviews',
                    'wrong_price',
                    'wrong_policy',
                    'missing_evidence',
                    'other'
                )
            );
    end if;

    if not exists (
        select 1 from pg_constraint where conname = 'chk_user_feedback_corrected_component'
    ) then
        alter table user_feedback
            add constraint chk_user_feedback_corrected_component
            check (
                corrected_component is null or corrected_component in (
                    'review_authenticity',
                    'seller_reliability',
                    'sentiment',
                    'return_policy_clarity',
                    'price_safety',
                    'user_feedback_history'
                )
            );
    end if;

    if not exists (
        select 1 from pg_constraint where conname = 'chk_user_feedback_expected_risk_level'
    ) then
        alter table user_feedback
            add constraint chk_user_feedback_expected_risk_level
            check (
                expected_risk_level is null or expected_risk_level in (
                    'Low Risk',
                    'Medium Risk',
                    'High Risk'
                )
            );
    end if;
end $$;

insert into model_versions (
    version,
    sentiment_model_name,
    fake_review_model_name,
    weights,
    metrics
)
values (
    '0.3.0',
    'distilbert-base-uncased-finetuned-sst-2-english',
    'calibrated_tfidf_fake_review_v3',
    '{
      "review_authenticity": 0.30,
      "seller_reliability": 0.20,
      "sentiment": 0.20,
      "return_policy_clarity": 0.15,
      "price_safety": 0.10,
      "user_feedback_history": 0.00
    }'::jsonb,
    '{"model_set": "v3_production", "feedback_scoring": "not_applied"}'::jsonb
)
on conflict (version) do nothing;
