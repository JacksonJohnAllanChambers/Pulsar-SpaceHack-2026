# Hand verdicts

The team looked at every contact the detector raised and said what it was. AIS can prove a contact is a
ship; only a person looking at the pixels can say one is not. These files are the ground truth behind
every precision figure in this repo.

| File | What |
| :-- | :-- |
| `us_labels.json` | 476 verdicts on the 16-scene US benchmark: 159 vessel, 287 not_vessel, 7 structure, 23 unsure (15 of them are `MISS_<scene>_<mmsi>` verdicts on AIS fixes with no contact) |
| `us_reference_contacts.json` | the 571 contacts of the run those verdicts were made on -- id, scene, pixel position only |
| `svalbard_labels.json` | 51 verdicts on the two Svalbard scenes: 8 vessel, 26 not_vessel, 17 unsure (2 are `MISS_*`) |
| `svalbard_reference_contacts.json` | the 49 contacts they were made on |

**Always pass `--reference`.** Detection ids renumber whenever the contact set changes, so matching
verdicts by id silently mislabels contacts. `scorecard.py --labels X --reference Y` carries each verdict
to the contact within 4 px of where it was given (`scripts/transfer_labels.py`); anything with no
labelled neighbour stays unlabelled and is excluded, never guessed. `MISS_*` verdicts are keyed by scene
and MMSI and carry over as they are.

`unsure` means a reviewer could not tell from 10 m pixels. It is excluded from precision in both
directions, and the count is always reported beside the figure.

The sheets the verdicts were made on are in `review/`; the procedure is `docs/LABELLING.md`.
