# Contact labelling — how the team adjudicates 589 real contacts

## Why we do this at all

The detector found 571 contacts across 16 real Sentinel-2 scenes. AIS tells us which of those are
broadcasting ships, and that gives us recall — but it cannot give us **precision**, because a contact
with no transponder is either a genuine unlisted vessel (which is the whole product) or a false alarm
(which is not). Only a human looking at the pixels can tell those apart.

Until someone does, our false-alarm number is an upper bound and we have to say so on stage. After,
it is a measurement. That is the entire point of this exercise, and it is worth an hour of the team's
time: **Long Beach's apparent 72.5 false alarms per 1000 km² turned out to be 26 once adjudicated.**

There are also 18 cards for the reverse case — AIS says a ship is in clear water and the detector
found nothing. Judge those too: if you can see a vessel, it is a real miss and it counts against us.
If you cannot, the transponder was lying or stale and it should not count. Be honest in both
directions; we do not want a number we cannot defend.

---

## One-time setup (Jack, ~2 minutes)

Without this, everyone labels into their own browser and has to export a file by hand. With it,
verdicts pool automatically and everyone sees the same progress.

1. Open <https://sheets.new> and name it **SpaceHack labels**.
2. **Extensions → Apps Script**, delete the stub, paste all of [`scripts/label_server.gs`](../scripts/label_server.gs), **Save**.
3. **Deploy → New deployment → Web app**, with:
   * **Execute as:** Me
   * **Who has access:** Anyone  ← required; teammates are not signed in to your Google account
4. Authorise when prompted, copy the `/exec` URL.
5. Paste it into [`review/config.js`](../review/config.js) and commit:

   ```js
   window.LABEL_ENDPOINT = "https://script.google.com/macros/s/AKfy.../exec";
   ```

The endpoint accepts anonymous writes by design. It holds vessel / not-vessel verdicts on public
Copernicus imagery and nothing else — treat it as throwaway hackathon infrastructure and delete the
deployment afterwards.

---

## For everyone else (30 seconds, then label)

```bash
git pull
```

Then **double-click the launcher** in the `review/` folder — it starts a local server, picks a free
port and opens your browser at the right page:

| | |
| :-- | :-- |
| Windows | `review\start-labelling.cmd` |
| macOS / Linux | `review/start-labelling.command` |

Nothing to install beyond Python itself — no packages, no virtualenv, no dataset. **Type your name
in the box at the top**, click your scene, and go. Keep the black window open while you label.

> Don't open the `.html` files directly. A `file://` page cannot reach the shared sheet, so your
> verdicts would silently stay on your laptop. The launcher exists precisely to stop that.

| Key | Verdict |
| :-- | :-- |
| **V** | a vessel — any boat, ship or craft, however small |
| **N** | not a vessel — cloud, whitecap, sandbar, pier, breakwater, wave texture, buoy wake |
| **S** | a fixed structure — oil platform, artificial island, pylon, channel marker |
| **U** | genuinely cannot tell — use it freely, `U` is excluded from the score rather than guessed |
| **← →** | move without deciding |

Each card shows the same patch four ways: RGB and NIR, close and wide. **The NIR close view is the
one that settles most calls** — water goes black in NIR and anything solid stays bright. The wide
views tell you whether the thing sits at the head of a wake (vessel) or is part of a shoal, pier or
cloud bank (not).

Cards already marked green arrived matched to a real AIS transponder, so they are vessels by
definition — skip them. You are there for the red **DARK VESSEL** cards.

Your verdicts save as you go and sync every couple of seconds; the indicator by your name says
`synced`, `N to sync`, or `offline`. **Going offline is safe** — verdicts queue in your browser and
flush when you reconnect, even across a reload.

---

## Who takes what

Everyone also does **Delaware (6) and San Francisco (11)** — 17 extra contacts, about two minutes.
Those two scenes are deliberately labelled by all of us so we can measure whether we actually agree
with each other. A precision figure resting on one person's unchecked judgement is just a different
kind of assumption, and this is exactly the thing a technical judge will poke at.

| Who | Scenes | Contacts |
| :-- | :-- | --: |
| Jack | ~~Long Beach~~, ~~Tampa~~ (done), Boston, Mississippi, Puget Sound, Savannah | 123 |
| 2nd | Miami, Charleston | 98 |
| 3rd | Galveston, Honolulu | 95 |
| 4th | San Diego, New York, Corpus Christi, Norfolk | 123 |
| **everyone** | **Delaware + San Francisco (overlap)** | **17 each** |

Fewer than four people? Hand the 4th column's scenes out between you — the split is advisory and the
shared sheet does not care who labels what. Miami and Tampa are the grim ones: shallow, turbid,
sandbar-ridden, and where the detector is weakest. That makes them the most valuable to get right.

---

## Pulling the results

```bash
python scripts/pull_labels.py                                   # -> data/outputs/review/labels.json
python scripts/scorecard.py -i data/real/s2_us_bundle --labels data/outputs/review/labels.json
```

`pull_labels.py` collapses the append-only sheet to one verdict per contact (latest wins, so
changing your mind works), reports how much each person did, and prints inter-rater agreement on
every contact two people judged — with the disagreements listed, so we can look at them together.

If the endpoint was not up when someone labelled, they can hit **Download labels.json** and we fold
their file in:

```bash
python scripts/pull_labels.py --merge ~/Downloads/labels.json
```

---

## Calling the hard ones

These are the disagreements worth pre-empting, drawn from the first 133 contacts:

* **Wake but no hull.** If there is a clean linear wake with a bright point at its head, it is a
  vessel, even when the hull is two pixels. That is the design working, not a false alarm.
* **Bright patch, no wake, in shallow water.** Usually a sandbar or shoal — `N`. Tampa and Miami are
  full of these.
* **Something at the very edge of the frame**, half black. Judge what you can see; if that is not
  enough, `U`.
* **A line of bright dots along a straight edge.** A pier, breakwater or jetty — `N`, or `S` if it is
  clearly a discrete built structure.
* **Three coloured dots in a row (red, green, blue).** An aircraft: the bands are exposed at slightly
  different instants. `N`. The detector already rejects most of these.
* **A wake with nothing at its head.** A vessel that has moved on, or a wake crossing the frame.
  Mark `U` unless you can see the craft.
