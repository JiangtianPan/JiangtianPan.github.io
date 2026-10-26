/**
 * City-level visit counter for jiangtianpan.github.io
 *
 * PRIVACY
 * -------
 * This Worker never reads the visitor's IP address. Cloudflare resolves
 * IP -> city at the edge *before* the request reaches this code, so all this
 * file ever touches is the already-resolved place name in `request.cf`.
 * `request.headers.get("CF-Connecting-IP")` is deliberately never called.
 *
 * What is stored is an aggregate counter per (day, city) — no per-visitor
 * rows, no session ids, no cookies, no user agent, no referrer, no path.
 * Two visits from the same person on the same day are indistinguishable from
 * two visits by different people.
 *
 * Deploy: see README.md in this directory.
 */

const ALLOWED_ORIGINS = ["https://jiangtianpan.github.io"];

function corsHeaders(origin) {
  return {
    // Echo the origin only when we actually trust it, so a stray site can't
    // read the response; the browser blocks it on mismatch.
    "Access-Control-Allow-Origin": ALLOWED_ORIGINS.includes(origin)
      ? origin
      : ALLOWED_ORIGINS[0],
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Max-Age": "86400",
  };
}

/** "CA" -> 🇨🇦 — regional indicator symbols are 0x1F1E6 + letter offset. */
function flag(cc) {
  if (!/^[A-Z]{2}$/.test(cc)) return "";
  return String.fromCodePoint(
    ...[...cc].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)
  );
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", {
        status: 405,
        headers: corsHeaders(origin),
      });
    }
    // Keeps the counter from being inflated by requests from anywhere else.
    if (!ALLOWED_ORIGINS.includes(origin)) {
      return new Response("Forbidden", {
        status: 403,
        headers: corsHeaders(origin),
      });
    }

    const cf = request.cf || {};
    const country = cf.country || "??";
    const region = cf.region || "";
    const city = cf.city || "";
    // Stored so a map can be drawn later without a second geo lookup. These
    // are Cloudflare's city-centre coordinates, not the visitor's position.
    const lat = cf.latitude ? Number(cf.latitude) : null;
    const lon = cf.longitude ? Number(cf.longitude) : null;

    const day = new Date().toISOString().slice(0, 10);

    // Referrer host, already stripped to a bare hostname by js/site.js. The
    // Referer header on THIS request is our own page, not the visitor's
    // origin, so the value has to come from the page via ?r=.
    let source = "";
    try {
      source = (new URL(request.url).searchParams.get("r") || "")
        .toLowerCase()
        .slice(0, 100);
      // Accept only something shaped like a hostname; anything else is junk
      // or an injection attempt, and is recorded as a direct visit.
      if (!/^[a-z0-9.-]+$/.test(source)) source = "";
    } catch (err) {
      source = "";
    }

    try {
      await env.DB.batch([
        env.DB.prepare(
          `INSERT INTO hits (day, country, region, city, lat, lon, count)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)
           ON CONFLICT (day, country, region, city)
           DO UPDATE SET count = count + 1`
        ).bind(day, country, region, city, lat, lon),
        env.DB.prepare(
          `INSERT INTO sources (day, source, count)
           VALUES (?1, ?2, 1)
           ON CONFLICT (day, source)
           DO UPDATE SET count = count + 1`
        ).bind(day, source),
      ]);
    } catch (err) {
      // A failed count must never surface to the visitor.
      console.error("insert failed:", err.message);
    }

    return new Response(null, { status: 204, headers: corsHeaders(origin) });
  },

  /** Cron Trigger: email yesterday's roll-up. Configured in wrangler.toml. */
  async scheduled(event, env) {
    const day = new Date(Date.now() - 86400000).toISOString().slice(0, 10);

    const [cityRes, sourceRes] = await env.DB.batch([
      env.DB.prepare(
        `SELECT country, region, city, count
           FROM hits
          WHERE day = ?1
          ORDER BY country ASC, count DESC`
      ).bind(day),
      env.DB.prepare(
        `SELECT source, count
           FROM sources
          WHERE day = ?1
          ORDER BY count DESC, source ASC`
      ).bind(day),
    ]);
    const results = cityRes.results || [];
    const sources = sourceRes.results || [];

    if (results.length === 0) {
      console.log(`no visits on ${day}, skipping email`);
      return;
    }

    const total = results.reduce((sum, r) => sum + r.count, 0);
    const cities = results.filter((r) => r.city).length;

    const byCountry = new Map();
    for (const r of results) {
      if (!byCountry.has(r.country)) byCountry.set(r.country, []);
      byCountry.get(r.country).push(r);
    }

    let rows = "";
    for (const [country, list] of byCountry) {
      const subtotal = list.reduce((s, r) => s + r.count, 0);
      // "??" is Cloudflare's placeholder when it cannot resolve the country.
      const label =
        country === "??" ? "Unknown" : `${flag(country)} ${escapeHtml(country)}`;
      rows +=
        `<tr><td colspan="2" style="padding:14px 0 4px;font-weight:600;` +
        `border-bottom:1px solid #e3e3df">${label}` +
        ` <span style="color:#71716a;font-weight:400">· ${subtotal}</span></td></tr>`;
      for (const r of list) {
        const place = [r.city, r.region].filter(Boolean).join(", ") || "Unknown";
        rows +=
          `<tr><td style="padding:4px 0;color:#46463f">${escapeHtml(place)}</td>` +
          `<td style="padding:4px 0;text-align:right;font-variant-numeric:tabular-nums;` +
          `color:#46463f">${r.count}</td></tr>`;
      }
    }

    // Sources block. Omitted entirely when every visit came in without a
    // referrer, since a lone "Direct / unknown" row says nothing.
    let sourceRows = "";
    if (sources.some((r) => r.source)) {
      sourceRows =
        `<tr><td colspan="2" style="padding:18px 0 4px;font-weight:600;` +
        `border-bottom:1px solid #e3e3df">Sources</td></tr>`;
      for (const r of sources) {
        const label = r.source ? escapeHtml(r.source) : "Direct / unknown";
        sourceRows +=
          `<tr><td style="padding:4px 0;color:#46463f">${label}</td>` +
          `<td style="padding:4px 0;text-align:right;font-variant-numeric:tabular-nums;` +
          `color:#46463f">${r.count}</td></tr>`;
      }
    }

    const html =
      `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;` +
      `max-width:420px;color:#1b1b19;line-height:1.5">` +
      `<p style="margin:0 0 2px;font-size:15px"><strong>${total}</strong> ` +
      `visit${total === 1 ? "" : "s"} from <strong>${cities}</strong> ` +
      `cit${cities === 1 ? "y" : "ies"}</p>` +
      `<p style="margin:0 0 16px;color:#71716a;font-size:13px">${day}</p>` +
      `<table style="width:100%;border-collapse:collapse;font-size:14px">${rows}${sourceRows}</table>` +
      `<p style="margin:22px 0 0;color:#8d8d86;font-size:11px;border-top:1px solid #e3e3df;` +
      `padding-top:10px">jiangtianpan.github.io · aggregate counts only, no IP addresses stored</p>` +
      `</div>`;

    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.RESEND_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: env.MAIL_FROM,
        to: env.MAIL_TO,
        subject: `Site visits · ${day} · ${total} from ${cities} ${cities === 1 ? "city" : "cities"}`,
        html,
      }),
    });

    if (!res.ok) {
      console.error(`resend failed ${res.status}: ${await res.text()}`);
    }
  },
};
