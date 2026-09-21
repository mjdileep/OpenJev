// Execute the unmodified original engine without rendering or random tile spawns.
const fs = require('fs'), vm = require('vm'), path = require('path');
for (const file of ['grid', 'tile', 'game_manager']) {
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, '../examples/game_2048/vendor', file + '.js'), 'utf8'));
}
const boards = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(boards.map(board => [0, 1, 2, 3].map(action => {
  const gm = Object.create(GameManager.prototype);
  Object.assign(gm, {size: 4, grid: new Grid(4), score: 0, over: false, won: false, keepPlaying: true});
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
    if (board[y][x]) gm.grid.insertTile(new Tile({x, y}, board[y][x]));
  }
  gm.actuate = () => {}; gm.addRandomTile = () => {};
  gm.move(action);
  return [Array.from({length: 4}, (_, y) => Array.from({length: 4}, (_, x) => gm.grid.cells[x][y]?.value || 0)), gm.score];
}))));
