create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists public.documents (
    id uuid primary key default gen_random_uuid(),
    filename text not null,
    status text not null default 'processing',
    page_count integer,
    chunk_count integer,
    uploaded_at timestamptz not null default timezone('utc', now()),
    processed_at timestamptz,
    metadata jsonb not null default '{}'::jsonb
);

create table if not exists public.document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references public.documents(id) on delete cascade,
    content text not null,
    embedding vector(384) not null,
    page_number integer not null,
    chunk_index integer not null,
    source_label text not null,
    tsv tsvector,
    created_at timestamptz not null default timezone('utc', now())
);

create index if not exists idx_document_chunks_document_id on public.document_chunks(document_id);
create index if not exists idx_document_chunks_embedding on public.document_chunks
using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists idx_document_chunks_tsv on public.document_chunks using gin(tsv);

create or replace function public.set_document_chunk_tsv()
returns trigger
language plpgsql
as $$
begin
    new.tsv := to_tsvector('english', coalesce(new.content, ''));
    return new;
end;
$$;

drop trigger if exists trg_set_document_chunk_tsv on public.document_chunks;
create trigger trg_set_document_chunk_tsv
before insert or update on public.document_chunks
for each row
execute function public.set_document_chunk_tsv();

create or replace function public.hybrid_search_chunks(
    query_text text,
    query_embedding vector(384),
    match_count integer default 8,
    document_ids uuid[] default null
)
returns table (
    id uuid,
    document_id uuid,
    filename text,
    content text,
    page_number integer,
    chunk_index integer,
    source_label text,
    vector_score double precision,
    text_score double precision,
    combined_score double precision
)
language sql
as $$
    with filtered as (
        select
            dc.id,
            dc.document_id,
            d.filename,
            dc.content,
            dc.page_number,
            dc.chunk_index,
            dc.source_label,
            1 - (dc.embedding <=> query_embedding) as vector_score,
            ts_rank_cd(dc.tsv, plainto_tsquery('english', query_text)) as text_score
        from public.document_chunks dc
        join public.documents d on d.id = dc.document_id
        where document_ids is null or dc.document_id = any(document_ids)
    )
    select
        id,
        document_id,
        filename,
        content,
        page_number,
        chunk_index,
        source_label,
        vector_score,
        text_score,
        (0.65 * coalesce(vector_score, 0) + 0.35 * coalesce(text_score, 0)) as combined_score
    from filtered
    order by combined_score desc, vector_score desc
    limit match_count;
$$;
