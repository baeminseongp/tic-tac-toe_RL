const X = 1;
const O = -1;
const EMPTY = 0;
const WIN_LINES = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
];

const boardEl = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const modelInfoEl = document.querySelector("#modelInfo");
const sideSelect = document.querySelector("#sideSelect");
const modeSelect = document.querySelector("#modeSelect");
const resetButton = document.querySelector("#resetButton");
const humanScoreEl = document.querySelector("#humanScore");
const drawScoreEl = document.querySelector("#drawScore");
const agentScoreEl = document.querySelector("#agentScore");

let model = null;
let board = Array(9).fill(EMPTY);
let human = X;
let agent = O;
let current = X;
let locked = true;
let scores = { human: 0, agent: 0, draw: 0 };

function encodeState(position, player) {
  const symbols = { "-1": "O", 0: ".", 1: "X" };
  return `${position.map((cell) => symbols[cell]).join("")}:${player === X ? "X" : "O"}`;
}

function legalActions(position) {
  return position.map((cell, index) => (cell === EMPTY ? index : -1)).filter((index) => index >= 0);
}

function winner(position) {
  for (const [a, b, c] of WIN_LINES) {
    const sum = position[a] + position[b] + position[c];
    if (sum === 3) return X;
    if (sum === -3) return O;
  }
  return EMPTY;
}

function terminalValue(position) {
  const wonBy = winner(position);
  if (wonBy !== EMPTY) return wonBy;
  if (position.every((cell) => cell !== EMPTY)) return 0;
  return null;
}

function probabilities(position, player, temperature = 0.05) {
  const actions = legalActions(position);
  const logits = model.logits[encodeState(position, player)] || Array(9).fill(0);
  const maxLogit = Math.max(...actions.map((action) => logits[action] / temperature));
  const weights = Array(9).fill(0);
  let total = 0;
  for (const action of actions) {
    const weight = Math.exp(logits[action] / temperature - maxLogit);
    weights[action] = weight;
    total += weight;
  }
  return weights.map((weight) => weight / total);
}

function chooseAgentAction() {
  const probs = probabilities(board, agent, modeSelect.value === "sample" ? 0.45 : 0.05);
  const actions = legalActions(board);
  if (modeSelect.value === "greedy") {
    return actions.reduce((best, action) => (probs[action] > probs[best] ? action : best), actions[0]);
  }
  let threshold = Math.random();
  for (const action of actions) {
    threshold -= probs[action];
    if (threshold <= 0) return action;
  }
  return actions[actions.length - 1];
}

function symbol(cell) {
  if (cell === X) return "X";
  if (cell === O) return "O";
  return "";
}

function render() {
  boardEl.innerHTML = "";
  board.forEach((cell, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `cell ${cell === X ? "x" : cell === O ? "o" : ""}`;
    button.textContent = symbol(cell);
    button.disabled = locked || cell !== EMPTY || current !== human;
    button.setAttribute("aria-label", `cell ${index + 1}`);
    button.addEventListener("click", () => playHuman(index));
    boardEl.appendChild(button);
  });
  humanScoreEl.textContent = scores.human;
  drawScoreEl.textContent = scores.draw;
  agentScoreEl.textContent = scores.agent;
}

function finishIfTerminal() {
  const value = terminalValue(board);
  if (value === null) return false;

  locked = true;
  if (value === 0) {
    scores.draw += 1;
    statusEl.textContent = "무승부입니다.";
  } else if (value === human) {
    scores.human += 1;
    statusEl.textContent = "당신이 이겼습니다.";
  } else {
    scores.agent += 1;
    statusEl.textContent = "모델이 이겼습니다.";
  }
  render();
  return true;
}

function playHuman(index) {
  if (locked || board[index] !== EMPTY || current !== human) return;
  board[index] = human;
  current = agent;
  render();
  if (!finishIfTerminal()) {
    locked = true;
    statusEl.textContent = "모델이 생각하는 중...";
    window.setTimeout(playAgent, 220);
  }
}

function playAgent() {
  if (terminalValue(board) !== null) return;
  const action = chooseAgentAction();
  board[action] = agent;
  current = human;
  locked = false;
  statusEl.textContent = "당신 차례입니다.";
  render();
  finishIfTerminal();
}

function resetGame() {
  human = sideSelect.value === "X" ? X : O;
  agent = human === X ? O : X;
  current = X;
  board = Array(9).fill(EMPTY);
  locked = model === null;
  statusEl.textContent = human === X ? "당신 차례입니다." : "모델이 먼저 둡니다.";
  render();
  if (model && agent === X) {
    locked = true;
    window.setTimeout(playAgent, 250);
  }
}

async function loadModel() {
  try {
    const response = await fetch("../model.json", { cache: "no-store" });
    model = await response.json();
    const episodes = model.episodes?.toLocaleString("ko-KR") || "?";
    modelInfoEl.textContent = `${episodes} 에피소드 self-play 학습 모델`;
    resetGame();
  } catch (error) {
    modelInfoEl.textContent = "model.json을 찾을 수 없습니다. 먼저 python3 train.py를 실행하세요.";
    statusEl.textContent = "모델 로드 실패";
    locked = true;
    render();
  }
}

sideSelect.addEventListener("change", resetGame);
modeSelect.addEventListener("change", resetGame);
resetButton.addEventListener("click", resetGame);

render();
loadModel();
