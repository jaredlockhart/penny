// Penny browser extension — background script

import { SERVER_URL } from "../protocol.js";

function init(): void {
  // TODO: establish WebSocket connection to Penny server
  console.log("Penny background script loaded", SERVER_URL);
}

init();
