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
