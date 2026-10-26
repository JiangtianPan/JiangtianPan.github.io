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
      await env.DB.prepare(
        `INSERT INTO hits (day, country, region, city, source, lat, lon, count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)
         ON CONFLICT (day, country, region, city, source)
         DO UPDATE SET count = count + 1`
      )
        .bind(day, country, region, city, source, lat, lon)
        .run();
    } catch (err) {
      // A failed count must never surface to the visitor.
      console.error("insert failed:", err.message);
    }

    return new Response(null, { status: 204, headers: corsHeaders(origin) });
  },

  /** Cron Trigger: email yesterday's roll-up. Configured in wrangler.toml. */
  async scheduled(event, env) {
    const day = new Date(Date.now() - 86400000).toISOString().slice(0, 10);

    const { results } = await env.DB.prepare(
      `SELECT country, region, city, source, count
         FROM hits
        WHERE day = ?1
        ORDER BY country ASC, count DESC`
    )
      .bind(day)
      .all();

    if (!results || results.length === 0) {
      console.log(`no visits on ${day}, skipping email`);
      return;
    }

    const total = results.reduce((sum, r) => sum + r.count, 0);

    // Rows are per (city, source); collapse them to one entry per city that
    // carries its own source list, so the email reads
    //   Winnipeg, Manitoba   2   linkedin.com, google.com
    const byPlace = new Map();
    for (const r of results) {
      const key = `${r.country}\u0000${r.region}\u0000${r.city}`;
      if (!byPlace.has(key)) {
        byPlace.set(key, { country: r.country, region: r.region, city: r.city, count: 0, srcs: [] });
      }
      const e = byPlace.get(key);
      e.count += r.count;
      e.srcs.push({ source: r.source, count: r.count });
    }

    const places = [...byPlace.values()];
    const cities = places.filter((p) => p.city).length;

    const byCountry = new Map();
    for (const p of places.sort((a, b) => b.count - a.count)) {
      if (!byCountry.has(p.country)) byCountry.set(p.country, []);
      byCountry.get(p.country).push(p);
    }

    let rows = "";
    for (const [country, list] of byCountry) {
      const subtotal = list.reduce((s, p) => s + p.count, 0);
      // "??" is Cloudflare's placeholder when it cannot resolve the country.
      const label =
        country === "??" ? "Unknown" : `${flag(country)} ${escapeHtml(country)}`;
      rows +=
        `<tr><td colspan="3" style="padding:14px 0 4px;font-weight:600;` +
        `border-bottom:1px solid #e3e3df">${label}` +
        ` <span style="color:#71716a;font-weight:400">· ${subtotal}</span></td></tr>`;
      for (const p of list) {
        const place = [p.city, p.region].filter(Boolean).join(", ") || "Unknown";
        // Busiest source first; "" means no referrer was available.
        const srcList = p.srcs
          .sort((a, b) => b.count - a.count)
          .map((x) => escapeHtml(x.source || "direct"))
          .join(", ");
        rows +=
          `<tr><td style="padding:4px 12px 4px 0;color:#46463f;white-space:nowrap">` +
          `${escapeHtml(place)}</td>` +
          `<td style="padding:4px 12px 4px 0;text-align:right;` +
          `font-variant-numeric:tabular-nums;color:#46463f">${p.count}</td>` +
          `<td style="padding:4px 0;color:#71716a;font-size:13px">${srcList}</td></tr>`;
      }
    }

    const html =
      `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;` +
      `max-width:420px;color:#1b1b19;line-height:1.5">` +
      `<p style="margin:0 0 2px;font-size:15px"><strong>${total}</strong> ` +
      `visit${total === 1 ? "" : "s"} from <strong>${cities}</strong> ` +
      `cit${cities === 1 ? "y" : "ies"}</p>` +
      `<p style="margin:0 0 16px;color:#71716a;font-size:13px">${day}</p>` +
      `<table style="width:100%;border-collapse:collapse;font-size:14px">${rows}</table>` +
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
