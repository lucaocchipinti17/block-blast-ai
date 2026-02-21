#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t FULL_ROW = 0xFFULL;
constexpr uint64_t FULL_COL_BASE = 0x0101010101010101ULL;
constexpr uint64_t LEFT_MASK = 0xFEFEFEFEFEFEFEFEULL;
constexpr uint64_t RIGHT_MASK = 0x7F7F7F7F7F7F7F7FULL;
constexpr uint64_t BOARD_MASK = 0xFFFFFFFFFFFFFFFFULL;

enum WeightIdx {
    W_BIG_L = 0,
    W_SQ3X3,
    W_SQ2X2,
    W_I5,
    W_I4,
    W_I3,
    W_LINE_CLEAR,
    W_EMPTY_ISLANDS,
    W_FILLED_ISLANDS,
    W_SMALL_ISLANDS,
    W_DENSITY,
    W_ROUGH_EDGES,
    W_COUNT
};

struct Step {
    int slot_idx;   // index within available arrays
    int row;
    int col;
    int lines;
};

struct Child {
    int slot_idx;
    int row;
    int col;
    int lines;
    uint64_t bits;
};

struct SearchResult {
    bool valid = false;
    double score = -1e18;
    std::vector<Step> steps;
};

struct StateKey {
    uint64_t bits;
    int depth;
    bool operator==(const StateKey& other) const {
        return bits == other.bits && depth == other.depth;
    }
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& k) const {
        return std::hash<uint64_t>{}(k.bits) ^ (std::hash<int>{}(k.depth) << 1);
    }
};

using Transposition = std::unordered_map<StateKey, SearchResult, StateKeyHash>;
using EvalCache = std::unordered_map<uint64_t, double>;

inline int popcount64(uint64_t x) {
    return static_cast<int>(__builtin_popcountll(x));
}

inline std::pair<uint64_t, int> place_piece(uint64_t bits, uint64_t shifted_mask) {
    bits |= shifted_mask;
    int lines = 0;

    for (int r = 0; r < 8; ++r) {
        uint64_t row_bits = (bits >> (r * 8)) & FULL_ROW;
        if (row_bits == FULL_ROW) {
            bits &= ~(FULL_ROW << (r * 8));
            ++lines;
        }
    }

    for (int c = 0; c < 8; ++c) {
        uint64_t col_mask = FULL_COL_BASE << c;
        if ((bits & col_mask) == col_mask) {
            bits &= ~col_mask;
            ++lines;
        }
    }

    return {bits, lines};
}

inline bool can_place(uint64_t board_bits, uint64_t shifted_mask) {
    return (board_bits & shifted_mask) == 0ULL;
}

inline uint64_t flood_expand(uint64_t seed, uint64_t allowed) {
    while (true) {
        uint64_t prev = seed;
        seed = (seed
                | (seed >> 8)
                | ((seed << 8) & BOARD_MASK)
                | ((seed >> 1) & LEFT_MASK)
                | ((seed << 1) & RIGHT_MASK)) & allowed;
        if (seed == prev) {
            break;
        }
    }
    return seed;
}

struct IslandStats {
    int count = 0;
    int small_count = 0; // size <= 3
};

inline IslandStats count_islands_stats(uint64_t target) {
    IslandStats stats;
    uint64_t remaining = target;
    while (remaining) {
        uint64_t seed = remaining & (~remaining + 1ULL);
        uint64_t region = flood_expand(seed, target);
        int size = popcount64(region);
        remaining &= ~region;
        if (size < 8) {
            ++stats.count;
            if (size <= 3) {
                ++stats.small_count;
            }
        }
    }
    return stats;
}

inline int rough_edges(uint64_t bits) {
    uint64_t up = bits >> 8;
    uint64_t down = bits << 8;
    uint64_t left = (bits >> 1) & LEFT_MASK;
    uint64_t right = (bits << 1) & RIGHT_MASK;
    uint64_t has_empty_neighbour = bits & ~(up & down & left & right) & BOARD_MASK;
    return popcount64(has_empty_neighbour);
}

inline std::vector<uint64_t> build_shifted_probe_masks(const std::vector<std::string>& rows) {
    const int h = static_cast<int>(rows.size());
    const int w = static_cast<int>(rows[0].size());

    uint64_t base_mask = 0ULL;
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            if (rows[r][c] == '1') {
                base_mask |= (1ULL << (r * 8 + c));
            }
        }
    }

    std::vector<uint64_t> out;
    out.reserve((8 - h + 1) * (8 - w + 1));
    for (int row = 0; row <= 8 - h; ++row) {
        for (int col = 0; col <= 8 - w; ++col) {
            int shift = row * 8 + col;
            out.push_back(base_mask << shift);
        }
    }
    return out;
}

struct Probes {
    std::vector<uint64_t> big_l_bl;
    std::vector<uint64_t> big_l_br;
    std::vector<uint64_t> big_l_tl;
    std::vector<uint64_t> big_l_tr;
    std::vector<uint64_t> sq3x3;
    std::vector<uint64_t> sq2x2;
    std::vector<uint64_t> i5h;
    std::vector<uint64_t> i5v;
    std::vector<uint64_t> i4h;
    std::vector<uint64_t> i4v;
    std::vector<uint64_t> i3h;
    std::vector<uint64_t> i3v;
};

const Probes& get_probes() {
    static const Probes probes{
        build_shifted_probe_masks({"100","100","111"}),
        build_shifted_probe_masks({"001","001","111"}),
        build_shifted_probe_masks({"111","100","100"}),
        build_shifted_probe_masks({"111","001","001"}),
        build_shifted_probe_masks({"111","111","111"}),
        build_shifted_probe_masks({"11","11"}),
        build_shifted_probe_masks({"11111"}),
        build_shifted_probe_masks({"1","1","1","1","1"}),
        build_shifted_probe_masks({"1111"}),
        build_shifted_probe_masks({"1","1","1","1"}),
        build_shifted_probe_masks({"111"}),
        build_shifted_probe_masks({"1","1","1"}),
    };
    return probes;
}

inline int count_fits(uint64_t bits, const std::vector<uint64_t>& masks) {
    int count = 0;
    for (uint64_t m : masks) {
        count += ((bits & m) == 0ULL) ? 1 : 0;
    }
    return count;
}

double evaluate_bits(
    uint64_t bits,
    const double* w,
    EvalCache& eval_cache,
    int eval_cache_max
) {
    auto it = eval_cache.find(bits);
    if (it != eval_cache.end()) {
        return it->second;
    }

    const Probes& p = get_probes();
    double score = 0.0;

    int bigl = count_fits(bits, p.big_l_bl)
             + count_fits(bits, p.big_l_br)
             + count_fits(bits, p.big_l_tl)
             + count_fits(bits, p.big_l_tr);
    score += bigl * (w[W_BIG_L] / 4.0);

    int sq3 = count_fits(bits, p.sq3x3);
    int sq2 = count_fits(bits, p.sq2x2);
    int i5 = count_fits(bits, p.i5h) + count_fits(bits, p.i5v);
    int i4 = count_fits(bits, p.i4h) + count_fits(bits, p.i4v);
    int i3 = count_fits(bits, p.i3h) + count_fits(bits, p.i3v);

    score += sq3 * w[W_SQ3X3];
    score += sq2 * w[W_SQ2X2];
    score += i5 * w[W_I5];
    score += i4 * w[W_I4];
    score += i3 * w[W_I3];

    IslandStats filled = count_islands_stats(bits);
    IslandStats empty = count_islands_stats((~bits) & BOARD_MASK);
    score += empty.count * w[W_EMPTY_ISLANDS];
    score += filled.count * w[W_FILLED_ISLANDS];
    score += empty.small_count * w[W_SMALL_ISLANDS];
    score += filled.small_count * w[W_SMALL_ISLANDS];

    score += popcount64(bits) * w[W_DENSITY];
    score += rough_edges(bits) * w[W_ROUGH_EDGES];

    if (eval_cache_max > 0 && static_cast<int>(eval_cache.size()) >= eval_cache_max) {
        eval_cache.clear();
    }
    eval_cache[bits] = score;
    return score;
}

std::vector<Child> generate_children(
    uint64_t board_bits,
    int slot_idx,
    uint64_t base_mask,
    int h,
    int w
) {
    std::vector<Child> out;
    out.reserve((8 - h + 1) * (8 - w + 1));
    for (int row = 0; row <= 8 - h; ++row) {
        for (int col = 0; col <= 8 - w; ++col) {
            int shift = row * 8 + col;
            uint64_t shifted = base_mask << shift;
            if (!can_place(board_bits, shifted)) {
                continue;
            }
            auto [new_bits, lines] = place_piece(board_bits, shifted);
            out.push_back({slot_idx, row, col, lines, new_bits});
        }
    }
    return out;
}

void limit_children(
    std::vector<Child>& children,
    int depth,
    bool use_sampling,
    int sample_size,
    int cap_depth1,
    int cap_depth2,
    std::mt19937_64& rng
) {
    if (children.empty()) {
        return;
    }

    if (use_sampling && depth == 0 && static_cast<int>(children.size()) > sample_size) {
        std::shuffle(children.begin(), children.end(), rng);
        children.resize(sample_size);
    }

    int cap = -1;
    if (depth == 1) cap = cap_depth1;
    if (depth == 2) cap = cap_depth2;
    if (cap <= 0 || static_cast<int>(children.size()) <= cap) {
        return;
    }

    std::vector<Child> clearers;
    std::vector<Child> non_clearers;
    clearers.reserve(children.size());
    non_clearers.reserve(children.size());
    for (const auto& c : children) {
        if (c.lines > 0) clearers.push_back(c);
        else non_clearers.push_back(c);
    }

    auto rank_fn = [](const Child& a, const Child& b) {
        if (a.lines != b.lines) return a.lines > b.lines;
        return popcount64(a.bits) < popcount64(b.bits);
    };
    std::sort(clearers.begin(), clearers.end(), rank_fn);
    std::sort(non_clearers.begin(), non_clearers.end(), rank_fn);

    std::vector<Child> limited;
    limited.reserve(cap);
    for (const auto& c : clearers) {
        if (static_cast<int>(limited.size()) >= cap) break;
        limited.push_back(c);
    }
    for (const auto& c : non_clearers) {
        if (static_cast<int>(limited.size()) >= cap) break;
        limited.push_back(c);
    }
    children.swap(limited);
}

struct SearchContext {
    const std::vector<uint64_t>& base_masks;
    const std::vector<int>& hs;
    const std::vector<int>& ws;
    const std::vector<int>& perm;
    const double* weights;
    bool use_sampling;
    int sample_size;
    int cap_depth1;
    int cap_depth2;
    int node_budget;
    bool has_deadline;
    std::chrono::steady_clock::time_point deadline;
    int eval_cache_max;

    int nodes = 0;
    bool exhausted = false;
    std::mt19937_64 rng;
    EvalCache& eval_cache;
    Transposition trans;
};

SearchResult dfs(SearchContext& ctx, uint64_t bits, int depth) {
    if (ctx.exhausted) {
        return {};
    }

    ++ctx.nodes;
    if (ctx.nodes >= ctx.node_budget) {
        ctx.exhausted = true;
        return {};
    }
    if (ctx.has_deadline && (ctx.nodes & 0xFF) == 0) {
        if (std::chrono::steady_clock::now() >= ctx.deadline) {
            ctx.exhausted = true;
            return {};
        }
    }

    StateKey key{bits, depth};
    auto it = ctx.trans.find(key);
    if (it != ctx.trans.end()) {
        return it->second;
    }

    if (depth == static_cast<int>(ctx.perm.size())) {
        SearchResult leaf;
        leaf.valid = true;
        leaf.score = evaluate_bits(bits, ctx.weights, ctx.eval_cache, ctx.eval_cache_max);
        ctx.trans.emplace(key, leaf);
        return leaf;
    }

    int slot = ctx.perm[depth];
    auto children = generate_children(
        bits,
        slot,
        ctx.base_masks[slot],
        ctx.hs[slot],
        ctx.ws[slot]
    );
    limit_children(
        children,
        depth,
        ctx.use_sampling,
        ctx.sample_size,
        ctx.cap_depth1,
        ctx.cap_depth2,
        ctx.rng
    );

    SearchResult best;
    for (const auto& child : children) {
        SearchResult child_res = dfs(ctx, child.bits, depth + 1);
        if (!child_res.valid) {
            continue;
        }
        double score = child_res.score + (child.lines * ctx.weights[W_LINE_CLEAR]);
        if (!best.valid || score > best.score) {
            best.valid = true;
            best.score = score;
            best.steps.clear();
            best.steps.reserve(child_res.steps.size() + 1);
            best.steps.push_back({child.slot_idx, child.row, child.col, child.lines});
            best.steps.insert(best.steps.end(), child_res.steps.begin(), child_res.steps.end());
        }
    }

    ctx.trans.emplace(key, best);
    return best;
}

SearchResult search_perm(
    uint64_t board_bits,
    const std::vector<uint64_t>& base_masks,
    const std::vector<int>& hs,
    const std::vector<int>& ws,
    const std::vector<int>& perm,
    const double* weights,
    bool use_sampling,
    int sample_size,
    int cap_depth1,
    int cap_depth2,
    int node_budget,
    bool has_deadline,
    std::chrono::steady_clock::time_point deadline,
    EvalCache& eval_cache,
    int eval_cache_max,
    int& nodes_used
) {
    std::random_device rd;
    SearchContext ctx{
        base_masks, hs, ws, perm, weights,
        use_sampling, sample_size, cap_depth1, cap_depth2,
        node_budget, has_deadline, deadline,
        eval_cache_max,
        0, false, std::mt19937_64(rd()), eval_cache, {}
    };
    SearchResult res = dfs(ctx, board_bits, 0);
    nodes_used = ctx.nodes;
    return res;
}

std::vector<Step> greedy_fallback(
    uint64_t board_bits,
    const std::vector<uint64_t>& base_masks,
    const std::vector<int>& hs,
    const std::vector<int>& ws,
    const std::vector<int>& available_slots,
    const double* weights,
    EvalCache& eval_cache,
    int eval_cache_max
) {
    std::vector<Step> plan;
    std::vector<int> remaining = available_slots;
    uint64_t cur_bits = board_bits;

    while (!remaining.empty()) {
        bool found = false;
        double best_score = -1e18;
        Step best_step{};
        uint64_t best_bits = cur_bits;

        for (int slot : remaining) {
            auto children = generate_children(cur_bits, slot, base_masks[slot], hs[slot], ws[slot]);
            for (const auto& ch : children) {
                double s = evaluate_bits(ch.bits, weights, eval_cache, eval_cache_max)
                         + (ch.lines * weights[W_LINE_CLEAR]);
                if (!found || s > best_score) {
                    found = true;
                    best_score = s;
                    best_step = {slot, ch.row, ch.col, ch.lines};
                    best_bits = ch.bits;
                }
            }
        }

        if (!found) {
            break;
        }

        plan.push_back(best_step);
        cur_bits = best_bits;
        remaining.erase(std::remove(remaining.begin(), remaining.end(), best_step.slot_idx), remaining.end());
    }

    return plan;
}

} // namespace

extern "C" int bb_best_plan(
    uint64_t board_bits,
    const uint64_t* base_masks,
    const int* hs,
    const int* ws,
    const int* piece_indices,
    int n_pieces,
    const double* weights,
    int sample_threshold,
    int sample_size,
    int time_budget_ms,
    int max_nodes,
    int cap_depth1,
    int cap_depth2,
    int eval_cache_max,
    int* out_len,
    int* out_piece_idx,
    int* out_row,
    int* out_col
) {
    if (!out_len || !out_piece_idx || !out_row || !out_col) {
        return 0;
    }
    *out_len = 0;
    if (n_pieces <= 0 || n_pieces > 3) {
        return 0;
    }

    std::vector<uint64_t> v_masks(base_masks, base_masks + n_pieces);
    std::vector<int> v_h(hs, hs + n_pieces);
    std::vector<int> v_w(ws, ws + n_pieces);
    std::vector<int> slots(n_pieces);
    for (int i = 0; i < n_pieces; ++i) {
        slots[i] = i;
    }

    std::vector<std::vector<int>> perms;
    {
        std::vector<int> p = slots;
        std::sort(p.begin(), p.end());
        do {
            perms.push_back(p);
        } while (std::next_permutation(p.begin(), p.end()));
    }

    auto estimate_first_count = [&](int slot) {
        int cnt = 0;
        int h = v_h[slot], w = v_w[slot];
        uint64_t base = v_masks[slot];
        for (int r = 0; r <= 8 - h; ++r) {
            for (int c = 0; c <= 8 - w; ++c) {
                uint64_t shifted = base << (r * 8 + c);
                cnt += can_place(board_bits, shifted) ? 1 : 0;
            }
        }
        return cnt;
    };

    std::vector<std::pair<int, std::vector<int>>> ranked;
    ranked.reserve(perms.size());
    int leaf_estimate = 0;
    for (const auto& p : perms) {
        int n = estimate_first_count(p[0]);
        ranked.push_back({n, p});
        int n2 = std::max(1, n / 2);
        int n4 = std::max(1, n / 4);
        leaf_estimate += n * n2 * n4;
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    bool use_sampling = leaf_estimate > sample_threshold;
    bool has_deadline = time_budget_ms > 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_budget_ms);
    int nodes_left = std::max(1, max_nodes);
    EvalCache eval_cache;

    SearchResult best;
    for (const auto& [_, perm] : ranked) {
        if (nodes_left <= 0) {
            break;
        }
        if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
            break;
        }

        int nodes_used = 0;
        SearchResult res = search_perm(
            board_bits,
            v_masks,
            v_h,
            v_w,
            perm,
            weights,
            use_sampling,
            sample_size,
            cap_depth1,
            cap_depth2,
            nodes_left,
            has_deadline,
            deadline,
            eval_cache,
            eval_cache_max,
            nodes_used
        );

        nodes_left -= nodes_used;
        if (res.valid && (!best.valid || res.score > best.score)) {
            best = std::move(res);
        }
    }

    std::vector<Step> final_steps;
    if (best.valid) {
        final_steps = best.steps;
    } else {
        final_steps = greedy_fallback(
            board_bits, v_masks, v_h, v_w, slots, weights, eval_cache, eval_cache_max
        );
    }

    if (final_steps.empty()) {
        return 0;
    }

    int out_n = std::min(static_cast<int>(final_steps.size()), n_pieces);
    *out_len = out_n;
    for (int i = 0; i < out_n; ++i) {
        const Step& s = final_steps[i];
        out_piece_idx[i] = piece_indices[s.slot_idx];
        out_row[i] = s.row;
        out_col[i] = s.col;
    }
    return 1;
}

