"use strict";
const $ = id => document.getElementById(id);
const directions = ["up", "right", "down", "left"];
const arrows = {up:"↑",right:"→",down:"↓",left:"←"};
let ready = false, running = false, pending = false, moves = 0, timer, metadata = {}, history = [];
class Input { on() {} }
class Storage {
  getGameState() { return null; } setGameState() {} clearGameState() {}
  getBestScore() { return 0; } setBestScore() {}
}
class View {
  continueGame() {}
  actuate(grid, data) {
    $("board").replaceChildren();
    let best = 0;
    for (let y=0;y<4;y++) for (let x=0;x<4;x++) {
      const value = grid.cells[x][y]?.value || 0;
      const tile = document.createElement("div");
      tile.className = "tile" + (value ? " filled" : "");
      tile.dataset.value = value; tile.textContent = value || "";
      tile.setAttribute("aria-label",`Row ${y+1}, column ${x+1}: ${value || "empty"}`);
      $("board").append(tile); best = Math.max(best, value);
    }
    $("score").textContent = data.score; $("best-tile").textContent = best;
    if (data.terminated) { running=false; $("status").textContent=data.won?"2048 reached!":"Game over"; }
  }
}
const game = new GameManager(4, Input, View, Storage);
function board() { return Array.from({length:4},(_,y)=>Array.from({length:4},(_,x)=>game.grid.cells[x][y]?.value||0)); }
function controls() {
  $("play").disabled=!ready||game.isGameTerminated();
  $("play").textContent=running?"Pause":"Play";
  $("step").disabled=!ready||pending||running||game.isGameTerminated();
  $("reset").disabled=pending;
  $("thinking").classList.toggle("active",pending);
}
function scores(values={},selected=null) {
  $("actions").replaceChildren();
  for (const action of directions) {
    const row=document.createElement("div");row.className="action-row"+(action===selected?" selected":"");
    const label=document.createElement("span");label.textContent=`${arrows[action]}  ${action[0].toUpperCase()+action.slice(1)}`;
    const track=document.createElement("div");track.className="track";
    const fill=document.createElement("div");fill.className="fill";fill.style.width=`${(values[action]||0)*100}%`;track.append(fill);
    const value=document.createElement("span");value.className="value";value.textContent=action in values?`${(values[action]*100).toFixed(1)}%`:selected?"illegal":"—";
    row.append(label,track,value);$("actions").append(row);
  }
}
function duration(seconds) { return seconds<1?`${Math.round(seconds*1000)} ms`:`${seconds.toFixed(2)} s`; }
async function step() {
  if(pending||!ready||game.isGameTerminated())return;
  pending=true;controls();$("error").hidden=true;$("status").textContent="Evaluating legal moves…";
  try {
    const before=board();
    const response=await fetch("/api/decide",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({board:before})});
    const data=await response.json();if(!response.ok)throw Error(data.error||"Decision failed");
    if(!directions.includes(data.action))throw Error("Unknown action from model");
    game.move(directions.indexOf(data.action));moves++;
    if(JSON.stringify(before)===JSON.stringify(board()))throw Error("Selected move did not change the board");
    data.after=board();data.score=game.score;history.push(data);
    $("action").textContent=`${arrows[data.action]} Swipe ${data.action}${data.forced?" · forced":""}`;
    scores(data.probabilities,data.action);$("latency").textContent=duration(data.usage.elapsed_seconds);
    $("average").textContent=duration(history.reduce((s,d)=>s+d.usage.elapsed_seconds,0)/history.length);
    $("batch").textContent=data.usage.candidate_batches.join(" + ");$("tokens").textContent=data.usage.generated_tokens;
    $("moves").textContent=`${moves} move${moves===1?"":"s"}`;
    if(data.peak_memory_gb)$("memory").textContent=`${data.peak_memory_gb.toFixed(2)} GB peak MLX`;
    $("inspect").textContent=JSON.stringify(data,null,2);$("export").disabled=false;
    const recent=history.slice(-24), max=Math.max(...recent.map(x=>x.usage.elapsed_seconds));$("chart").replaceChildren();
    for(const item of recent){const bar=document.createElement("div");bar.style.height=`${Math.max(4,item.usage.elapsed_seconds/max*100)}%`;bar.title=duration(item.usage.elapsed_seconds);$("chart").append(bar);}
    $("status").textContent=game.isGameTerminated()?(game.won?"2048 reached!":"Game over"):running?"Playing locally":"Paused";
  } catch(error) { running=false;$("error").hidden=false;$("error").textContent=error.message;$("status").textContent="Stopped"; }
  finally {pending=false;controls();if(running&&!game.isGameTerminated())timer=setTimeout(step,Number($("pace").value));}
}
$("play").onclick=()=>{running=!running;clearTimeout(timer);controls();if(running)step();else $("status").textContent=pending?"Pausing after this move…":"Paused";};
$("step").onclick=step;
$("reset").onclick=()=>{running=false;clearTimeout(timer);history=[];moves=0;$("error").hidden=true;$("tokens").textContent="0";$("memory").textContent="Live inference";game.restart();scores();$("action").textContent="Ready to play";$("moves").textContent="0 moves";for(const id of ["latency","average","batch"])$(id).textContent="—";$("chart").replaceChildren();$("inspect").textContent="Play a move to inspect it.";$("export").disabled=true;$("status").textContent=ready?"Ready":"Loading model…";controls();};
$("export").onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify({metadata,history},null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="openjev-2048-run.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
scores();
async function poll() {
  try {const response=await fetch("/api/status");metadata=await response.json();if(metadata.error)throw Error(metadata.error);
    $("model").textContent=metadata.model;$("hardware").textContent=`${metadata.backend?metadata.backend.toUpperCase()+" · ":""}${metadata.hardware}`;
    if(metadata.ready){ready=true;$("status").textContent="Ready · local inference";controls();return;}
  }catch(error){$("error").hidden=false;$("error").textContent=error.message;}
  setTimeout(poll,2000);
}poll();
