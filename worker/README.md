# City-level visit counter

A Cloudflare Worker that counts visits to jiangtianpan.github.io by city and
emails a daily roll-up. Deployed separately from the site itself — GitHub Pages
serves only static files and cannot run this.

## What it stores, and what it doesn't

It never reads the visitor's IP. Cloudflare resolves IP → city at the edge
before the request arrives, so the Worker only ever sees the resolved place
name in `request.cf`. `CF-Connecting-IP` is never called.

The database holds one aggregate row per `(day, city)`:

| day | country | region | city | lat | lon | count |
|-----|---------|--------|------|-----|-----|-------|
| 2026-10-11 | CA | Ontario | Toronto | 43.70 | -79.42 | 7 |

No per-visitor rows, no session ids, no cookies, no user agent, no referrer.
Two visits by one person are indistinguishable from visits by two people.

## Deploy

Requires a free Cloudflare account and `npm i -g wrangler`.

Note: these blocks carry no inline `#` comments on purpose. zsh does not treat
`#` as a comment in an interactive shell (`interactive_comments` is off by
default), so a pasted comment is passed to the command as arguments.

**1. Log in**

```sh
cd worker
wrangler login
```

**2. Register a workers.dev subdomain** — one-time, per Cloudflare account, and
only doable in the browser. A brand new account has none, and `wrangler deploy`
fails with "You need to register a workers.dev subdomain" until it exists:

    https://dash.cloudflare.com/<account-id>/workers/onboarding

The subdomain becomes the host of every Worker on the account, so pick
something you are happy to expose publicly.

**3. Create the database** and copy the printed `database_id` into
`wrangler.toml`:

```sh
wrangler d1 create visits --location enam
```

`--location enam` keeps the primary in Eastern North America. Drop or change it
if you are elsewhere.

**4. Create the table:**

```sh
wrangler d1 execute visits --remote --file=./schema.sql
```

**5. Deploy.** The Resend key is only read by the daily cron, so counting works
without it — you can deploy now and add the key later:

```sh
wrangler deploy
wrangler secret put RESEND_API_KEY
```

`wrangler deploy` prints the Worker URL. Put it in `ANALYTICS_ENDPOINT` at the
top of `../js/site.js` — until then the site sends no requests at all.

## Email

Register at [resend.com](https://resend.com) **with the address in `MAIL_TO`**.
Resend's shared sender `onboarding@resend.dev` needs no domain of your own, but
it will only deliver to the address that owns the Resend account; anything else
returns 403. To send elsewhere later, verify a domain and change `MAIL_FROM`.

Free tiers in play: Workers 100k req/day, D1 100k writes/day, Resend 3000
emails/month. A personal site will not come close to any of them.

## Excluding your own visits

Load any page with `#toggle-goatcounter` appended:

    https://jiangtianpan.github.io/#toggle-goatcounter

An alert confirms it. That flips `localStorage.skipgc` in this browser, which
GoatCounter's own `count.js` checks before sending — and `../js/site.js` reads
the same flag before pinging this Worker, so the one toggle silences both.
Load the same URL again to turn counting back on.

Per browser and per device, and lost if you clear site data. There is no
server-side exclusion: the Worker would need to recognise you, and it has
nothing to recognise you by — which is the point.

## Checking it works

```sh
wrangler tail                                  # live request log
wrangler d1 execute visits --remote \
  --command="SELECT * FROM hits ORDER BY day DESC LIMIT 20"
echo '{"scheduled":true}'                      # or trigger the cron by hand:
wrangler dev --test-scheduled                  # then curl /__scheduled
```

Nothing is emailed on days with zero visits.

## Later: a map

`lat`/`lon` are already stored, so a map needs no schema change and no second
geo lookup — query the table, feed the points to Leaflet. Prefer aggregating to
country or region level for anything shown publicly on the site.
