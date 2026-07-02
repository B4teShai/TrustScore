-- Audit + review-sample tables used by scan persistence. These existed in
-- db/schema.sql but were never added as a migration, so databases built from
-- migrations alone failed every scan write with "relation does not exist".

create table if not exists prediction_run_audit (
    prediction_run_id uuid primary key references prediction_runs(id) on delete cascade,
    requested_target_market text,
    resolved_market text,
    resolved_country text,
    locale text,
    language text,
    product_snapshot jsonb not null default '{}'::jsonb,
    evidence jsonb not null default '[]'::jsonb,
    missing_inputs jsonb not null default '[]'::jsonb,
    score_trace jsonb not null default '[]'::jsonb,
    model_modes jsonb not null default '{}'::jsonb,
    model_versions jsonb not null default '{}'::jsonb,
    model_artifact_status jsonb not null default '{}'::jsonb,
    recommendation_source text,
    market_reference jsonb not null default '{}'::jsonb,
    extraction_profile jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_prediction_run_audit_market_created
    on prediction_run_audit(resolved_market, created_at desc);
create index if not exists idx_prediction_run_audit_country_created
    on prediction_run_audit(resolved_country, created_at desc);

create table if not exists scan_review_samples (
    id uuid primary key default gen_random_uuid(),
    prediction_run_id uuid not null references prediction_runs(id) on delete cascade,
    review_hash text not null,
    redacted_text text,
    rating double precision check (rating is null or (rating >= 0 and rating <= 5)),
    verified_purchase boolean,
    review_date text,
    source_position integer check (source_position is null or source_position >= 0),
    retention_class text not null default 'short_lived',
    created_at timestamptz not null default now()
);

create index if not exists idx_scan_review_samples_run_id
    on scan_review_samples(prediction_run_id);

-- Migration 003's constraint predates four issue categories the API accepts
-- (wrong_page_type, wrong_market, wrong_currency, wrong_extracted_field);
-- rebuild it with the full list so feedback with those categories persists.
alter table user_feedback drop constraint if exists chk_user_feedback_issue_category;
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
            'wrong_page_type',
            'wrong_market',
            'wrong_currency',
            'wrong_extracted_field',
            'missing_evidence',
            'other'
        )
    );
