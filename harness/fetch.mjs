// GET a URL with Node's fetch and print the body. DuckDuckGo serves a bot challenge to Python/curl's
// TLS fingerprint but not to Node's, so bgtools.http_get shells out to this for search.
const [url, limit = "600000"] = process.argv.slice(2);
const r = await fetch(url, {
  headers: { "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0", Accept: "text/html" },
  redirect: "follow", signal: AbortSignal.timeout(20000),
});
process.stdout.write((await r.text()).slice(0, Number(limit)));
