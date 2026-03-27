// Penny browser extension — background script
// Manages WebSocket connection to Penny server and browser tool API

const SERVER_URL = "ws://localhost:9090";

let ws = null;
let connected = false;

function connect() {
  // TODO: establish WebSocket connection to Penny server
  console.log("Penny background script loaded");
}

connect();
