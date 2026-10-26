-- City-level visit counts for jiangtianpan.github.io
--
-- One row per (day, city). There is no per-visitor row and no identifier of
-- any kind, so an individual visit cannot be isolated or traced back.
--
-- region/city are NOT NULL DEFAULT '' on purpose: they are part of the primary
-- key, and in SQLite NULL never equals NULL, which would silently break the
-- ON CONFLICT upsert and insert a duplicate row on every single visit.

CREATE TABLE IF NOT EXISTS hits (
  day     TEXT    NOT NULL,
  country TEXT    NOT NULL,
  region  TEXT    NOT NULL DEFAULT '',
  city    TEXT    NOT NULL DEFAULT '',
  lat     REAL,
  lon     REAL,
  count   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, country, region, city)
);

CREATE INDEX IF NOT EXISTS idx_hits_day ON hits (day);

-- Where visitors came from, as a bare hostname ("linkedin.com"), never a full
-- URL. A referrer URL can carry the visitor's identity in its path — e.g.
-- linkedin.com/in/some-person — so js/site.js strips everything but the host
-- before sending, and nothing here can be traced back to an individual.
--
-- Deliberately a separate table rather than more columns on `hits`: folding
-- source into that primary key would multiply rows by city x source, while
-- the daily email wants the two dimensions summarised independently.
--
-- source = '' means no referrer: a direct visit, a link opened from an app or
-- a PDF, or a site sending no referrer header. Those are indistinguishable.

CREATE TABLE IF NOT EXISTS sources (
  day    TEXT    NOT NULL,
  source TEXT    NOT NULL DEFAULT '',
  count  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, source)
);

CREATE INDEX IF NOT EXISTS idx_sources_day ON sources (day);
