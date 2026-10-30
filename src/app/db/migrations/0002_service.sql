-- DECISION-26: what the P6 service needs beyond PRD §11.2.

-- What to process (file path or URL) and the finished-job stats the result JSON needs.
ALTER TABLE jobs ADD COLUMN input JSONB NOT NULL DEFAULT '{}';
ALTER TABLE jobs ADD COLUMN stats JSONB;
-- Leases (FR-6.2): how many times a worker has claimed the job.
ALTER TABLE jobs ADD COLUMN attempts INT NOT NULL DEFAULT 0;
-- Strict FIFO even when two jobs share a created_at.
ALTER TABLE jobs ADD COLUMN seq BIGSERIAL;
-- REAL is float32: 0.4 would read back as 0.4000000059604645.
ALTER TABLE jobs ALTER COLUMN progress TYPE DOUBLE PRECISION;
CREATE INDEX jobs_pickup ON jobs (status, seq);

-- Ordered progress events for SSE (FR-6.3); a reconnecting client resumes after its last id.
CREATE TABLE job_events (
  id BIGSERIAL PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES jobs,
  ts TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  stage TEXT NOT NULL,
  progress DOUBLE PRECISION NOT NULL,
  message TEXT NOT NULL
);
CREATE INDEX job_events_by_job ON job_events (job_id, id);

-- Results are fetched by job; summary_nodes are ordered as the pipeline produced them.
ALTER TABLE summary_nodes ADD COLUMN ord INT NOT NULL DEFAULT 0;
CREATE INDEX summary_nodes_by_job ON summary_nodes (job_id, ord);
CREATE INDEX key_points_by_job ON key_points (job_id, rank);
ALTER TABLE claims ADD COLUMN ord INT NOT NULL DEFAULT 0;
CREATE INDEX claims_by_job ON claims (job_id, ord);
