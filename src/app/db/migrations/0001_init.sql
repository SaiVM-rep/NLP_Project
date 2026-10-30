-- PRD §11.2, verbatim. Never edit an applied migration: add a new file.
CREATE TABLE documents (
  doc_id UUID PRIMARY KEY,
  source TEXT NOT NULL,
  title TEXT,
  full_text TEXT NOT NULL,
  structure JSONB NOT NULL,          -- sections, paragraphs, sentences as spans
  structure_ok BOOLEAN NOT NULL DEFAULT TRUE,
  token_count INT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE jobs (
  job_id UUID PRIMARY KEY,
  doc_id UUID REFERENCES documents,
  status TEXT NOT NULL,              -- queued|running|done|failed
  stage TEXT,
  progress REAL DEFAULT 0,
  route TEXT,
  config JSONB NOT NULL,
  error TEXT,
  locked_by TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE summary_nodes (
  id UUID PRIMARY KEY,
  job_id UUID REFERENCES jobs,
  level INT NOT NULL,
  parent_id UUID,
  text TEXT NOT NULL,
  source_chunk_ids JSONB
);

CREATE TABLE key_points (
  id UUID PRIMARY KEY,
  job_id UUID REFERENCES jobs,
  rank INT, text TEXT, cluster_size INT,
  ev_start INT, ev_end INT, similarity REAL
);

CREATE TABLE claims (
  id UUID PRIMARY KEY,
  job_id UUID REFERENCES jobs,
  summary_node_id UUID,
  sentence_idx INT, claim TEXT,
  evidence JSONB,
  deberta_entail REAL, deberta_contra REAL, minicheck_support REAL,
  label_deberta TEXT, label_minicheck TEXT
);

CREATE TABLE llm_calls (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT, job_id UUID,
  ts TIMESTAMPTZ DEFAULT now(),
  provider TEXT, model TEXT, purpose TEXT,
  prompt_hash TEXT, tokens_in INT, tokens_out INT,
  latency_ms INT, retries INT, cache_hit BOOLEAN, error TEXT
);

CREATE TABLE eval_runs (
  run_id TEXT PRIMARY KEY,
  experiment TEXT, git_commit TEXT, config JSONB, started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ
);
