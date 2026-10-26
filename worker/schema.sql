-- City + referrer visit counts for jiangtianpan.github.io
--
-- One row per (day, city, source). There is no per-visitor row and no
-- identifier of any kind, so an individual visit cannot be isolated or
-- traced back.
--
-- region/city/source are NOT NULL DEFAULT '' on purpose: they are part of the
-- primary key, and in SQLite NULL never equals NULL, which would silently
-- break the ON CONFLICT upsert and insert a duplicate row on every visit.
--
-- `source` holds only the referrer's HOSTNAME, never a full URL. A referrer
-- URL can identify a person through its path — linkedin.com/in/their-name —
-- so js/site.js strips everything but the host before sending, and the Worker
-- rejects anything that is not hostname-shaped. '' means no referrer was
-- available: a direct visit, a link opened from an app or a PDF, or a site
-- sending no referrer header. Those cases are indistinguishable.

CREATE TABLE IF NOT EXISTS hits (
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

CREATE INDEX IF NOT EXISTS idx_hits_day ON hits (day);
