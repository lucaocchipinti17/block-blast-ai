# Block Blast AI — How the Model and Testing Work

## What Are We Building?

If you've played Block Blast, you know the loop: you get three pieces, place them wherever they fit on the 8×8 grid, and earn points when you complete a full row or column. The longer you keep clearing lines on consecutive turns, the bigger your streak multiplier.

This document explains how a computer agent learns to play Block Blast — not by memorising patterns, but by thinking ahead and scoring every possible sequence of moves.

---

## The Files at a Glance

| File | What it does |
|------|-------------|
| `board.py` | The 8×8 board — placing pieces, clearing lines |
| `pieces.py` | Every piece shape the game can give you |
| `game.py` | Game loop, scoring, streak logic |
| `model.py` | The AI agent — searches moves, scores board states |
| `test.py` | Runs the agent through full games and records results |
| `stream_capture.py` | Window-located live capture loop (mss + title lookup) |
| `piece_bank_detector.py` | Classical CV detector for 3-slot piece bank -> 5x5 matrices |
| `live_cv_bridge.py` | Capture + detect + heuristic planning orchestration |
| `piece_bank_cv.py` | Computer-vision piece bank detector (3 inventory slots from a frame) |

---

## Quick Start (New Machine)

Recommended Python: `3.11` or `3.12`.

```bash
cd /path/to/blockblast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the GUI:

```bash
python round_planner_gui.py
```

---

## Runtime Config (`runtime_config.json`)

Machine-specific live-CV settings are now in:

`runtime_config.json`

Default shape:

```json
{
  "live_window_title_contains": "Movie Recording",
  "live_bank_roi_norm_xyxy": [0.0633, 0.6945, 0.9221, 0.8135],
  "expected_cell_px": 20.0,
  "capture_target_fps": 20.0,
  "capture_window_refresh_sec": 1.0,
  "live_trigger_keybind": "<F8>"
}
```

If someone else runs this project, they should edit only this file for:
- window title matching
- normalized bank ROI
- known piece cell size (currently 20 px)
- capture loop timing
- hotkey for live capture trigger

---

## Calibration / Verification

Before first live run on a new machine, verify ROI splitting with:

```bash
python live_roi_debug.py --title "Movie Recording" --roi "0.0633,0.6945,0.9221,0.8135"
```

Expected result:
- yellow rectangle matches the full bank ROI
- three slot rectangles align left/middle/right piece regions
- slot previews at the bottom show each captured region correctly

Then in the GUI:
1. Click `Attach Stream`
2. Press `F8` each turn (`capture -> recognize -> evaluate -> display`)  
   Keybind is configurable in `runtime_config.json` via `live_trigger_keybind`.

---

## Live CV Automation Pipeline

The production path for mirrored-stream automation is:

1. `stream_capture.WindowCaptureLoop` continuously captures a target desktop window by title.
2. `piece_bank_detector.PieceBankDetector` reads the configured bank ROI, splits into 3 slots, segments piece cells, and converts each slot to a 5x5 matrix.
3. `live_cv_bridge.LiveCVPlanner` feeds those piece matrices to `HeuristicAgent` and returns move order + placements.
4. `round_planner_gui.py` displays move previews and applies the final board state.  
   Press `F8` in GUI after attaching stream to trigger capture -> recognize -> evaluate -> display.

GUI live controls:
- `Window title contains` (e.g. `Movie Recording`)
- `Bank ROI norm x0,y0,x1,y1` (normalized coordinates in captured window)
- `Attach Stream`, then press `F8` for each cycle
- `runtime_config.json` supports `live_trigger_keybind` (example: `<F8>` or `<Return>`)

Default ROI normalization in GUI is derived from approximate calibration:
- Window approx: `(1028,25)` to `(1439,899)`
- Bank ROI approx: `(1054,632)` to `(1407,736)`
- Converted to normalized ROI and clamped robustly for proportional resizing.

---

## CV Piece Bank Reader (`piece_bank_cv.py`)

Use this module to interpret the 3 pieces shown in a phone stream screenshot.

Install dependency:

```bash
pip install -r requirements.txt
```

Run on a screenshot:

```bash
python piece_bank_cv.py --image /path/to/frame.png --show
```

If default slot regions miss your UI layout, pass explicit ROIs (`x,y,w,h`) for
left/middle/right slots:

```bash
python piece_bank_cv.py \
  --image /path/to/frame.png \
  --roi 80,1260,300,420 \
  --roi 420,1260,300,420 \
  --roi 760,1260,300,420 \
  --show
```

Interactive mouse ROI test (recommended for QuickTime phone mirroring):

```bash
python piece_bank_cv_test.py --screen
```

Workflow:
- Hover the cursor over the full bank's top-left corner and press `Space`.
- Hover over the bottom-right corner and press `Space` again.
- Press `R` to reset corner selection, `Esc`/`Q` to cancel.
- The script splits that region into 3 horizontal slot ROIs.
- It prints the detected piece for each slot as a 5×5 array in terminal.

Continuous scanning mode with same selected ROI:

```bash
python piece_bank_cv_test.py --screen --watch --interval 0.75
```

---

## How the Board Works (`board.py`)

The board is an 8×8 grid. Each cell is either **empty (0)** or **filled (1)**. When you place a piece, its cells are written onto the board. Then the board checks every row and every column — if any of them are fully filled, they're cleared and you score points.

```
Before:                  After placing █ at (7,0):
┌────────┐               ┌────────┐
│········│               │········│
│········│               │········│
│·███████│  +  █   →     │········│   ← row 7 cleared!
└────────┘               └────────┘
```

Pieces are stored as 5×5 arrays with empty padding around them. Before anything is placed, the code strips the padding to find the real footprint — the tightest bounding box around the filled cells. This means the AI only checks the cells that actually matter.

---

## How Scoring Works (`game.py`)

The score formula was reverse-engineered from real game data:

```
points = tiles_placed + (lines_cleared × streak × 10)
```

A few things to notice:

- **Tiles placed** always count — even a move that clears nothing scores 1 point per cell placed.
- **Line clear bonus** scales linearly with the number of lines cleared in that move.
- **Streak window**: a streak only continues if the next line clear happens within **3 moves** of the previous clear.
- **Streak reset behavior**: if you go more than 3 moves without a line clear, streak drops to 0; the next clear starts streak at 1.

---

## How the Board is Stored Internally (Bitwise Representation)

Rather than looking up a 2D array for every check, the agent stores the entire board state as a **single 64-bit integer**. Each bit represents one cell:

```
bit 0  = row 0, col 0  (top-left)
bit 1  = row 0, col 1
...
bit 63 = row 7, col 7  (bottom-right)
```

This makes the three most common operations essentially instantaneous:

| Operation | How it works |
|-----------|-------------|
| **Does piece overlap?** | `board & piece_mask == 0` — one bitwise AND |
| **Place a piece** | `board \|= piece_mask` — one bitwise OR |
| **Is row 7 full?** | `(board >> 56) & 0xFF == 0xFF` — one shift and compare |

Every piece shape is pre-converted to a bitmask once at startup. When the agent wants to try placing a piece at position (row=3, col=4), it just looks up the pre-computed mask for that position — no loops, no array indexing.

---

## How the Agent Thinks (`model.py`)

### The Core Idea: Think 3 Pieces Ahead

At each turn, the agent doesn't just ask "where's the best place for this piece?" It asks: **"What's the best sequence for all 3 pieces in my hand?"**

This matters because the order you place pieces in changes what's possible. Placing the big 3×3 block first might lock out a spot you needed for the L-piece. So the agent tries all possible orderings.

### Step 1 — Generate All Orderings

With 3 pieces, there are 6 possible orderings (1→2→3, 1→3→2, 2→1→3, etc.). The agent considers all of them.

### Step 2 — Search Every Placement Sequence

For each ordering, it builds a tree of possibilities:

```
Ordering: [Piece A, Piece B, Piece C]

Place A at pos (0,0) → board state 1
  Place B at pos (2,3) → board state 1a
    Place C at pos (4,1) → LEAF: score this final board
    Place C at pos (4,5) → LEAF: score this final board
  Place B at pos (3,1) → board state 1b
    ...
Place A at pos (0,2) → board state 2
  ...
```

Every leaf node is a possible end state after all 3 pieces are placed. The agent scores all of them and picks the sequence that leads to the highest score.

### Step 3 — Sampling for Large Searches

On a full board, there might be hundreds of valid positions for each piece, leading to millions of leaf nodes. When the estimated leaf count exceeds 50,000, the agent switches to **random sampling** — instead of trying every position for the first piece, it randomly samples 100 of them. This is a deliberate trade-off: slightly worse decisions in exchange for not taking 10 minutes per move.

---

## How the Agent Scores a Board (`evaluate_board`)

After placing all 3 pieces, the agent scores the resulting board state. The score has two parts: **rewards** and **penalties**.

### Rewards — Things the Agent Wants to See

**Piece-fit flexibility** is the most important concept. Instead of just looking at the current board, the agent asks: *how many future pieces could fit on this board?*

It probes the board with several shapes and counts how many positions each one fits into empty space:

| Shape | Why it matters | Weight |
|-------|---------------|--------|
| Big-L corners (all 4 rotations) | Open corners = highly flexible board | ×10 each |
| 3×3 square | Most space-efficient shape — fits lots of pieces | ×20 |
| 2×2 square | Smaller version of same idea | ×5 |
| Straight lines (3, 4, 5 cells) | Rows/columns about to clear | ×0.5–2 |

A board with many 3×3 fits is one where almost any piece can land somewhere useful. A board with zero 3×3 fits is getting dangerously cluttered.

**Line clears** are also rewarded directly — every line cleared in the sequence of 3 placements adds 30 points to the heuristic score.

### Penalties — Things the Agent Wants to Avoid

| Problem | Description | Weight |
|---------|-------------|--------|
| Empty islands | Isolated pockets of empty space that no piece can fill | −5 each |
| Filled islands | Isolated clumps of filled cells that can't be cleared | −10 each |
| Tiny regions (size 1–3) | Pockets so small literally nothing fits — game-ending traps | −20 each |
| Board density | Too many filled cells overall (mild pressure) | −0.5 per cell |
| Rough edges | Jagged filled/empty boundaries (harder to clear) | −0.5 per boundary |

**Islands** are found using a flood-fill algorithm — same idea as the paint bucket tool. Starting from any unfound cell, it expands outward to all connected cells of the same type. If the resulting region is small and disconnected from the rest, it's flagged.

This is also implemented with bitwise operations: instead of looping over a 2D array, the flood fill expands by shifting the current region's bitmask up, down, left, and right simultaneously, masking out edge wrapping, and iterating until stable.

---

## How the Tests Work (`test.py`)

`test.py` runs the agent through complete games with no human input, then reports results.

### What Happens in a Test Game

```
1. Draw 3 random pieces
2. Check if any can be placed → if not, game over
3. Ask the agent for its best move
4. Place the piece, clear lines, update score and streak
5. Repeat until all 3 pieces placed
6. If >3 moves pass without a clear, streak resets
7. Go to step 1
```

### Running Tests

```bash
# 10 games (default)
python test.py

# 50 games
python test.py --runs 50

# See every placement the agent makes
python test.py --runs 5 --verbose

# See agent search diagnostics (leaf counts, sampling mode, scores)
python test.py --runs 5 --agent-verbose
```

### Example Output

```
  Block Blast — Heuristic Agent Test
  Runs: 10
  ────────────────────────────────────────────────────────────
   Run     Score   Turns   Rounds      Time
  ────────────────────────────────────────────────────────────
     1      4821      87       29    12.34s
     2      3102      63       21     9.71s
     3      6540     112       37    17.88s
    ...
  ────────────────────────────────────────────────────────────
          Score   Turns
  High     6540     112
  Low      3102      63
  Avg      4821    87.3              13.21s
  ────────────────────────────────────────────────────────────

  High score: 6540
  Total time: 132.10s
```

---

## What the Agent Does Well (and Doesn't)

**What it does well:**
- Keeps the board clean and avoids creating isolated pockets
- Recognises when a board state has many future placement options
- Naturally avoids game-over positions by preferring open boards

**What it doesn't account for:**
- **Streak value** — the agent doesn't know that its current streak of 5 makes the next line worth 6× more than usual. A smarter version would bias heavily toward any move that maintains the streak.
- **Future piece distribution** — it doesn't know which pieces are more likely to appear next. A 3×3 opening on an otherwise clean board is statistically more useful than an L-shaped one.
- **Piece ordering within a round** — the agent commits to one complete 3-piece sequence. In reality you can adapt after each placement since you see the same 3 pieces throughout the round.

These limitations are exactly what a reinforcement learning approach would be designed to overcome — but even with these constraints, the heuristic agent should significantly outperform average human play.

---

## Tuning the Agent

All heuristic weights live in `DEFAULT_WEIGHTS` at the top of `model.py`. You can override them when creating the agent:

```python
agent = HeuristicAgent(weights={
    "sq3x3":      40.0,   # value open space more
    "line_clear": 50.0,   # chase lines harder
    "density":    -1.0,   # penalise clutter more
})
```

To tune these systematically, the next step would be an evolutionary algorithm: run many games with slightly varied weights, keep the weights that produce the highest scores, and repeat. Because the board is small and games are short, this can run thousands of iterations on a laptop overnight.
