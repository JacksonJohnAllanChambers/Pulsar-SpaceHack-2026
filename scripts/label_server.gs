/**
 * Shared label store for the contact review pages (Google Apps Script web app).
 *
 * Why Apps Script: the review pages are static HTML that teammates open from the repo, but
 * precision has to be pooled across everyone. This gives a write endpoint with no GCP project,
 * no billing, no API keys in the page and nothing for a teammate to install -- and the verdicts
 * land in a Sheet the team can watch fill up in real time.
 *
 * DEPLOY (about two minutes, once):
 *   1. https://sheets.new  -> name it "SpaceHack labels"
 *   2. Extensions > Apps Script, delete the stub, paste this file, Save
 *   3. Deploy > New deployment > type "Web app"
 *        Execute as:      Me
 *        Who has access:  Anyone            <- required: teammates are not signed in to your Google account
 *   4. Authorise when prompted, copy the /exec URL
 *   5. Put that URL in review/config.js and commit it
 *
 * The endpoint takes anonymous writes by design, so treat it as throwaway hackathon
 * infrastructure: it holds vessel/not-vessel verdicts on public Copernicus imagery and nothing
 * else. Do not reuse it for anything you would mind a stranger writing to, and delete the
 * deployment when the hackathon is over.
 *
 * POST body (text/plain, to dodge a CORS preflight): {labeller, verdicts: [{id, verdict}]}
 * GET  ?since=<iso>  -> {rows: [{id, verdict, labeller, ts}], count}
 */

var SHEET_NAME = 'labels';
var HEADERS = ['ts', 'labeller', 'id', 'verdict'];

function sheet_() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) {
    sh = ss.insertSheet(SHEET_NAME);
    sh.appendRow(HEADERS);
    sh.setFrozenRows(1);
  }
  return sh;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    lock.waitLock(20000);
    var body = JSON.parse(e.postData.contents);
    var labeller = String(body.labeller || 'anonymous').slice(0, 60);
    var verdicts = body.verdicts || [];
    if (!verdicts.length) return json_({ok: true, written: 0});

    // Append-only: every verdict is a row, and the newest row for an id wins on read. Keeps
    // concurrent labellers from clobbering each other and leaves an audit trail of changed minds.
    var now = new Date().toISOString();
    var rows = verdicts.slice(0, 2000).map(function (v) {
      return [now, labeller, String(v.id).slice(0, 80), String(v.verdict).slice(0, 20)];
    });
    sheet_().getRange(sheet_().getLastRow() + 1, 1, rows.length, HEADERS.length).setValues(rows);
    return json_({ok: true, written: rows.length});
  } catch (err) {
    return json_({ok: false, error: String(err)});
  } finally {
    try { lock.releaseLock(); } catch (ignored) {}
  }
}

function doGet(e) {
  try {
    var sh = sheet_();
    var last = sh.getLastRow();
    if (last < 2) return json_({rows: [], count: 0});
    var values = sh.getRange(2, 1, last - 1, HEADERS.length).getValues();
    var since = (e && e.parameter && e.parameter.since) || '';
    var rows = [];
    for (var i = 0; i < values.length; i++) {
      var ts = values[i][0] instanceof Date ? values[i][0].toISOString() : String(values[i][0]);
      if (since && ts <= since) continue;
      rows.push({ts: ts, labeller: values[i][1], id: values[i][2], verdict: values[i][3]});
    }
    return json_({rows: rows, count: rows.length});
  } catch (err) {
    return json_({rows: [], count: 0, error: String(err)});
  }
}
