alter table products add column if not exists normalized_url text;
update products set normalized_url = split_part(url, '#', 1) where normalized_url is null;
alter table products alter column normalized_url set not null;
alter table products add column if not exists product_image_url text;
alter table products add column if not exists return_policy text;

create unique index if not exists idx_products_normalized_url on products(normalized_url);

alter table prediction_runs add column if not exists browser_id_hash text;
alter table prediction_runs add column if not exists fetch_mode text;
alter table prediction_runs add column if not exists extraction_signals jsonb not null default '[]'::jsonb;

alter table user_feedback add column if not exists browser_id_hash text;
alter table user_feedback drop column if exists browser_id;

create index if not exists idx_prediction_runs_product_created_at
    on prediction_runs(product_id, created_at desc);
create index if not exists idx_prediction_runs_browser_created_at
    on prediction_runs(browser_id_hash, created_at desc)
    where browser_id_hash is not null;
create index if not exists idx_user_feedback_prediction_browser
    on user_feedback(prediction_run_id, browser_id_hash)
    where browser_id_hash is not null;

do $$
begin
    if not exists (
        select 1 from pg_constraint where conname = 'chk_sellers_rating_range'
    ) then
        alter table sellers
            add constraint chk_sellers_rating_range
            check (rating is null or (rating >= 0 and rating <= 5));
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_sellers_review_count_nonnegative'
    ) then
        alter table sellers
            add constraint chk_sellers_review_count_nonnegative
            check (review_count is null or review_count >= 0);
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_sellers_years_active_nonnegative'
    ) then
        alter table sellers
            add constraint chk_sellers_years_active_nonnegative
            check (years_active is null or years_active >= 0);
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_products_price_nonnegative'
    ) then
        alter table products
            add constraint chk_products_price_nonnegative
            check (price is null or price >= 0);
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_products_average_market_price_positive'
    ) then
        alter table products
            add constraint chk_products_average_market_price_positive
            check (average_market_price is null or average_market_price > 0);
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_products_rating_range'
    ) then
        alter table products
            add constraint chk_products_rating_range
            check (rating is null or (rating >= 0 and rating <= 5));
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_products_review_count_nonnegative'
    ) then
        alter table products
            add constraint chk_products_review_count_nonnegative
            check (review_count is null or review_count >= 0);
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_reviews_rating_range'
    ) then
        alter table reviews
            add constraint chk_reviews_rating_range
            check (rating is null or (rating >= 0 and rating <= 5));
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_prediction_runs_trust_score_range'
    ) then
        alter table prediction_runs
            add constraint chk_prediction_runs_trust_score_range
            check (trust_score >= 0 and trust_score <= 100);
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_prediction_runs_risk_level'
    ) then
        alter table prediction_runs
            add constraint chk_prediction_runs_risk_level
            check (risk_level in ('Low Risk', 'Medium Risk', 'High Risk'));
    end if;
    if not exists (
        select 1 from pg_constraint where conname = 'chk_prediction_runs_confidence_range'
    ) then
        alter table prediction_runs
            add constraint chk_prediction_runs_confidence_range
            check (confidence is null or (confidence >= 0 and confidence <= 1));
    end if;
end $$;
