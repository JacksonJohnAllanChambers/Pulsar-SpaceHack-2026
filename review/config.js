// Shared label store for the contact review pages.
//
// Deploy scripts/label_server.gs as a Google Apps Script web app (Execute as: Me,
// Who has access: Anyone), then paste its /exec URL here and commit this file.
// Leave it empty and the pages still work -- verdicts just stay in your own browser
// and have to be exported with "Download labels.json".
// Deployment access must be "Anyone", not "Anyone with a Google account": the pages fetch this
// cross-origin from 127.0.0.1, and the account-restricted setting answers with a login redirect
// that CORS then blocks, so every teammate would just see "offline".
window.LABEL_ENDPOINT = "https://script.google.com/macros/s/AKfycbzd8dMeCHloQTtrWj2KnzzdRspWtapV0_SPX2TG0kkfr3ffHGeRZPCeauQnSZdRKnJi1A/exec";
