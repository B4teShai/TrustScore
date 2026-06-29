create extension if not exists "pgcrypto";

create table if not exists sellers (
    id uuid primary key default gen_random_uuid(),
    name text,
    rating double precision,
    review_count integer,
    years_active integer,
    created_at timestamptz not null default now()
);

create table if not exists products (
    id uuid primary key default gen_random_uuid(),
    seller_id uuid references sellers(id) on delete set null,
    url text not null,
    site text,
    title text,
    description text,
    price double precision,
    currency text,
    average_market_price double precision,
    rating double precision,
    review_count integer,
    created_at timestamptz not null default now()
);

create index if not exists idx_products_url on products(url);
create index if not exists idx_products_site on products(site);

create table if not exists reviews (
    id uuid primary key default gen_random_uuid(),
    product_id uuid not null references products(id) on delete cascade,
    text text not null,
    rating double precision,
    verified_purchase boolean,
    review_date text,
    created_at timestamptz not null default now()
);

create index if not exists idx_reviews_product_id on reviews(product_id);

create table if not exists model_versions (
    id uuid primary key default gen_random_uuid(),
    version text not null unique,
    sentiment_model_name text,
    fake_review_model_name text,
    weights jsonb,
    metrics jsonb,
    created_at timestamptz not null default now()
);

create table if not exists prediction_runs (
    id uuid primary key default gen_random_uuid(),
    product_id uuid not null references products(id) on delete cascade,
    model_version_id uuid references model_versions(id) on delete set null,
    trust_score integer not null,
    risk_level text not null,
    confidence double precision,
    component_scores jsonb not null,
    top_reasons jsonb not null,
    recommendation text,
    created_at timestamptz not null default now()
);

create index if not exists idx_prediction_runs_product_id on prediction_runs(product_id);
create index if not exists idx_prediction_runs_created_at on prediction_runs(created_at);

create table if not exists model_predictions (
    id uuid primary key default gen_random_uuid(),
    prediction_run_id uuid not null references prediction_runs(id) on delete cascade,
    model_name text not null,
    input_features jsonb,
    output jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_model_predictions_run_id on model_predictions(prediction_run_id);

create table if not exists user_feedback (
    id uuid primary key default gen_random_uuid(),
    prediction_run_id uuid not null references prediction_runs(id) on delete cascade,
    browser_id text,
    helpful boolean not null,
    comment text,
    created_at timestamptz not null default now()
);

create index if not exists idx_user_feedback_prediction_run_id
    on user_feedback(prediction_run_id);
