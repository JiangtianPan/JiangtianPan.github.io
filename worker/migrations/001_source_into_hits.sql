-- Fold `source` into the hits primary key, so the daily email can show which
-- sources a given city came from. Replaces the separate `sources` table.
--
-- SQLite cannot alter a primary key, so the table is rebuilt. Existing rows
-- predate referrer tracking and get source = '' (unknown).
--
-- Run once:  wrangler d1 execute visits --remote --file=./migrations/001_source_into_hits.sql

CREATE TABLE IF NOT EXISTS hits_new (
  day     TEXT    NOT NULL,
  country TEXT    NOT NULL,
  region  TEXT    NOT NULL DEFAULT '',
  city    TEXT    NOT NULL DEFAULT '',
  source  TEXT    NOT NULL DEFAULT '',
  lat     REAL,
  lon     REAL,
  count   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, country, region, city, source)
);

INSERT INTO hits_new (day, country, region, city, source, lat, lon, count)
  SELECT day, country, region, city, '', lat, lon, count FROM hits;

DROP TABLE hits;
ALTER TABLE hits_new RENAME TO hits;
CREATE INDEX IF NOT EXISTS idx_hits_day ON hits (day);

DROP TABLE IF EXISTS sources;
