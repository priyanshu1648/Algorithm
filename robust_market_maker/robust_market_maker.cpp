#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::vector;

// ---------------------------------------------------------------------------
// Market simulator
// ---------------------------------------------------------------------------
struct MarketCfg {
    double S0 = 100.0;
    double T = 1.0;
    int    N = 2000;                 // steps per day
    double sigma = 2.0;              // base volatility (price per sqrt(day))
    double stress_mult = 2.5;        // vol multiplier in stress regime
    double p_enter_stress = 0.0008;
    double p_exit_stress = 0.004;
    double k_normal = 1.5;           // depth decay, calm market
    double k_stress = 0.9;           // orders reach deeper when stressed
    double A = 140.0;                // market orders per side per day
    double flow_beta = 0.5;          // informed flow tilt
    double alpha_std = 6.0;          // std of hidden drift (price per day)
    double alpha_theta = 20.0;       // mean reversion of hidden drift
    double jump_rate = 3.0;          // jumps per day
    double jump_std = 1.5;
    int    latency = 2;              // quotes are based on a price this many steps old
    double fee = 0.005;              // per unit traded
    double liq_cost = 0.5;           // per unit cost of crossing the spread
    double tick = 0.01;
    double base_size = 3.0;          // max quoted size per side per step
};

struct Day {
    vector<double> S;        // N+1 mid prices
    vector<int> regime;      // N
    vector<int> nb, ns;      // buy / sell market order counts per step
    vector<double> reach;    // depth reached by each order, flat
    vector<int> off;         // N+1 offsets into reach
};

// One day of price path + order tape. Independent of any strategy, so every
// strategy can be replayed on exactly the same day.
Day generate_day(std::mt19937_64& rng, const MarketCfg& m) {
    int N = m.N;
    double dt = m.T / N;
    double sq = std::sqrt(dt);
    double nu = std::sqrt(2.0 * m.alpha_theta) * m.alpha_std;

    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    Day d;
    d.S.assign(N + 1, 0.0);
    d.regime.assign(N, 0);
    d.nb.assign(N, 0);
    d.ns.assign(N, 0);
    d.off.assign(N + 1, 0);
    vector<double> alpha(N);

    d.S[0] = m.S0;
    double a = m.alpha_std * norm(rng);
    bool stress = false;
    for (int t = 0; t < N; t++) {
        double u = uni(rng);
        if (stress) {
            if (u < m.p_exit_stress) stress = false;
        } else if (u < m.p_enter_stress) {
            stress = true;
        }
        d.regime[t] = stress ? 1 : 0;
        alpha[t] = a;
        double vol = m.sigma * (stress ? m.stress_mult : 1.0);
        double jump = 0.0;
        if (uni(rng) < m.jump_rate * dt) jump = m.jump_std * norm(rng);
        d.S[t + 1] = d.S[t] + a * dt + vol * sq * norm(rng) + jump;
        a = a * (1.0 - m.alpha_theta * dt) + nu * sq * norm(rng);
    }

    // informed traders: buy/sell counts tilt with the hidden drift
    int total = 0;
    for (int t = 0; t < N; t++) {
        double tilt = m.flow_beta * std::tanh(alpha[t] / m.alpha_std);
        std::poisson_distribution<int> pb(m.A * (1.0 + tilt) * dt);
        std::poisson_distribution<int> ps(m.A * (1.0 - tilt) * dt);
        d.nb[t] = pb(rng);
        d.ns[t] = ps(rng);
        total += d.nb[t] + d.ns[t];
    }
    d.reach.reserve(total);
    for (int t = 0; t < N; t++) {
        double k = d.regime[t] ? m.k_stress : m.k_normal;
        std::exponential_distribution<double> ex(k);
        int n = d.nb[t] + d.ns[t];
        d.off[t + 1] = d.off[t] + n;
        for (int i = 0; i < n; i++) d.reach.push_back(ex(rng));
    }
    return d;
}

// ---------------------------------------------------------------------------
// Strategy interface
// ---------------------------------------------------------------------------
struct Quote {
    double bid, ask, bid_size, ask_size, flat;   // flat > 0 means sell that many units
};

struct Strategy {
    std::string name;
    virtual ~Strategy() {}
    virtual void reset() {}
    virtual void update(double s_obs, const double* buys, int nbuy,
                        const double* sells, int nsell) {}
    virtual Quote quote(int t, double s, double q, double eq) = 0;
};

// ---------------------------------------------------------------------------
// Baseline: textbook Avellaneda-Stoikov (fixed inputs, finite horizon)
// ---------------------------------------------------------------------------
struct VanillaAS : Strategy {
    const MarketCfg& m;
    double gamma, sigma, k, dt;

    VanillaAS(const MarketCfg& mc, double g = 0.1, double sg = 2.0, double kk = 1.5)
        : m(mc), gamma(g), sigma(sg), k(kk), dt(mc.T / mc.N) {
        name = "Vanilla A-S";
    }

    Quote quote(int t, double s, double q, double eq) override {
        double tau = m.T - t * dt;
        double var = sigma * sigma;
        double r = s - q * gamma * var * tau;
        double half = 0.5 * gamma * var * tau + std::log(1.0 + gamma / k) / gamma;
        double bid = std::floor((r - half) / m.tick) * m.tick;
        double ask = std::ceil((r + half) / m.tick) * m.tick;
        return {bid, ask, m.base_size, m.base_size, 0.0};
    }
};

// ---------------------------------------------------------------------------
// Upgraded version
// ---------------------------------------------------------------------------
struct RobustAS : Strategy {
    const MarketCfg& m;
    double gamma, h, q_soft, q_hard, dd_limit;
    bool use_flow, use_limits, use_guard;
    double dt;

    double a_v, a_k, flow_decay, a_b, pred_cap, c_tox, c_lat, min_edge;
    int H, jump_cd, dd_cd;

    // regression state is kept across days (it is a learned parameter)
    double cov = 0.0, varm = 1.0;

    // per-day state
    double var = 0.0, k_inv = 0.0, m_flow = 0.0, peak_eq = 0.0, pred = 0.0;
    bool has_prev = false;
    double prev_s = 0.0;
    int cooldown = 0;
    std::deque<std::pair<double, double>> hist;

    RobustAS(const MarketCfg& mc, const std::string& nm, bool flow = true,
             bool limits = true, bool guard = true, double g = 0.1,
             double h_roll = 0.3, double qs = 10.0, double qh = 18.0, double dd = 25.0)
        : m(mc), gamma(g), h(h_roll), q_soft(qs), q_hard(qh), dd_limit(dd),
          use_flow(flow), use_limits(limits), use_guard(guard), dt(mc.T / mc.N) {
        name = nm;
        a_v = 2.0 / (40 + 1);              // vol EWMA (about 40 steps)
        a_k = 0.02;                        // k EWMA (per order)
        flow_decay = std::exp(-dt / 0.03);
        H = 100;                           // prediction horizon in steps
        a_b = 1.0 / 300;                   // regression EWMA rate
        pred_cap = 1.0;                    // max price shift from the predictor
        c_tox = 0.25;                      // widening per unit of |prediction|
        c_lat = 1.0;                       // latency buffer multiplier
        min_edge = 0.02;
        jump_cd = 40;
        dd_cd = 150;
    }

    void reset() override {
        var = m.sigma * m.sigma;
        k_inv = 1.0 / m.k_normal;
        m_flow = 0.0;
        hist.clear();
        has_prev = false;
        cooldown = 0;
        peak_eq = 0.0;
        pred = 0.0;
    }

    void update(double s_obs, const double* buys, int nbuy,
                const double* sells, int nsell) override {
        // robust volatility: big moves are winsorized so one jump does not
        // blow up the estimate, but they do trigger cooldown mode
        if (has_prev) {
            double ret = s_obs - prev_s;
            double sd = std::sqrt(var * dt);
            if (std::fabs(ret) > 6 * sd) {
                if (use_guard) cooldown = std::max(cooldown, jump_cd);
                ret = std::copysign(6 * sd, ret);
            }
            var += a_v * (ret * ret / dt - var);
        }
        prev_s = s_obs;
        has_prev = true;

        // k from the tape: depth of market orders is exponential(k), MLE = 1/mean
        for (int i = 0; i < nbuy; i++) k_inv += a_k * (buys[i] - k_inv);
        for (int i = 0; i < nsell; i++) k_inv += a_k * (sells[i] - k_inv);

        // signed order flow with decay
        m_flow = m_flow * flow_decay + (nbuy - nsell);
        hist.push_back({m_flow, s_obs});
        if ((int)hist.size() > H + 1) hist.pop_front();
        if ((int)hist.size() == H + 1) {
            double m0 = hist.front().first, s0 = hist.front().second;
            double dS = s_obs - s0;
            cov += a_b * (m0 * dS - cov);
            varm += a_b * (m0 * m0 - varm);
        }
        double beta = cov / (varm + 1e-6);
        pred = std::max(-pred_cap, std::min(pred_cap, beta * m_flow));
    }

    Quote quote(int t, double s, double q, double eq) override {
        double tfrac = t * dt;
        // risk aversion ramps up in the last 10% of the day so we end flat
        double gam = gamma * (1.0 + 4.0 * std::max(0.0, (tfrac - 0.9 * m.T) / (0.1 * m.T)));
        double k = std::min(3.0, std::max(0.5, 1.0 / k_inv));
        double p = use_flow ? pred : 0.0;

        double r = s + p - q * gam * var * h;
        double half = 0.5 * gam * var * h + std::log(1.0 + gamma / k) / gamma;
        half += c_tox * std::fabs(p);
        half += c_lat * std::sqrt(var * dt * m.latency);   // stale quote buffer
        half = std::max(half, m.fee + min_edge);

        double size_b = m.base_size, size_a = m.base_size;
        if (use_guard) {
            peak_eq = std::max(peak_eq, eq);
            if (peak_eq - eq > dd_limit) {
                cooldown = std::max(cooldown, dd_cd);
                peak_eq = eq;
            }
            if (cooldown > 0) {
                half *= 1.8;                 // stay visible, but wider and smaller
                size_b *= 0.5;
                size_a *= 0.5;
                cooldown -= 1;
            }
        }

        double flat = 0.0;
        if (use_limits) {
            size_b *= std::max(0.0, 1.0 - std::max(q, 0.0) / q_soft);    // long -> buy less
            size_a *= std::max(0.0, 1.0 - std::max(-q, 0.0) / q_soft);   // short -> sell less
            if (std::fabs(q) > q_hard) flat = std::copysign(std::fabs(q) - q_hard, q);
        }

        double bid = std::floor((r - half) / m.tick) * m.tick;
        double ask = std::ceil((r + half) / m.tick) * m.tick;
        return {bid, ask, size_b, size_a, flat};
    }
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
struct DayResult {
    double pnl, max_dd, avg_abs_inv, max_inv, volume;
    double avg_spread, stress_spread, uptime, stress_uptime;
};

DayResult run_day(const Day& day, Strategy& strat, const MarketCfg& m) {
    int N = m.N, lat = m.latency;
    strat.reset();

    double q = 0, cash = 0, peak = 0, maxdd = 0;
    double volume = 0, spread_sum = 0, stress_sum = 0, inv_abs = 0, max_inv = 0;
    int two_sided = 0, stress_two = 0, stress_steps = 0;
    const double* prev_b = nullptr;
    const double* prev_s = nullptr;
    int prev_nb = 0, prev_ns = 0;

    for (int t = 0; t < N; t++) {
        double s_true = day.S[t];
        double s_obs = day.S[std::max(t - lat, 0)];   // we only see a delayed price
        double eq = cash + q * s_true;
        peak = std::max(peak, eq);
        maxdd = std::max(maxdd, peak - eq);

        strat.update(s_obs, prev_b, prev_nb, prev_s, prev_ns);
        Quote qt = strat.quote(t, s_obs, q, eq);
        double bid = qt.bid, ask = qt.ask, bsz = qt.bid_size, asz = qt.ask_size;
        if (qt.flat != 0.0) {                          // forced flatten, pays the spread
            cash += qt.flat * s_true - std::fabs(qt.flat) * (m.liq_cost + m.fee);
            q -= qt.flat;
        }

        bool stressed = day.regime[t] == 1;
        if (stressed) stress_steps++;
        if (bsz > 0 && asz > 0) {
            two_sided++;
            spread_sum += ask - bid;
            if (stressed) {
                stress_two++;
                stress_sum += ask - bid;
            }
        }

        int lo = day.off[t], mid = day.off[t] + day.nb[t], hi = day.off[t + 1];
        const double* buys = day.reach.data() + lo;
        const double* sells = day.reach.data() + mid;
        int nbuy = mid - lo, nsell = hi - mid;

        double d_ask = ask - s_true;                   // depth is measured from the TRUE mid
        for (int i = 0; i < nbuy; i++) {               // market buys lift our ask
            if (asz > 0 && buys[i] > d_ask) {
                double f = std::min(1.0, asz);
                asz -= f;
                cash += (ask - m.fee) * f;
                q -= f;
                volume += f;
            }
        }
        double d_bid = s_true - bid;
        for (int i = 0; i < nsell; i++) {              // market sells hit our bid
            if (bsz > 0 && sells[i] > d_bid) {
                double f = std::min(1.0, bsz);
                bsz -= f;
                cash -= (bid + m.fee) * f;
                q += f;
                volume += f;
            }
        }

        inv_abs += std::fabs(q);
        max_inv = std::max(max_inv, std::fabs(q));
        prev_b = buys; prev_nb = nbuy;
        prev_s = sells; prev_ns = nsell;
    }

    double pnl = cash + q * day.S[N] - std::fabs(q) * (m.liq_cost + m.fee);  // liquidate at close
    double nan = std::nan("");
    DayResult r;
    r.pnl = pnl;
    r.max_dd = maxdd;
    r.avg_abs_inv = inv_abs / N;
    r.max_inv = max_inv;
    r.volume = volume;
    r.avg_spread = two_sided ? spread_sum / two_sided : nan;
    r.stress_spread = stress_two ? stress_sum / stress_two : nan;
    r.uptime = (double)two_sided / N;
    r.stress_uptime = stress_steps ? (double)stress_two / stress_steps : nan;
    return r;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
double mean_of(const vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double sample_std(const vector<double>& v) {
    double m = mean_of(v), ss = 0.0;
    for (double x : v) ss += (x - m) * (x - m);
    return std::sqrt(ss / (v.size() - 1));
}

double nanmean(const vector<double>& v) {
    double s = 0.0;
    int n = 0;
    for (double x : v) if (!std::isnan(x)) { s += x; n++; }
    return n ? s / n : std::nan("");
}

struct Summary {
    double mean_pnl, std_pnl, sharpe, cvar5, worst_day, max_dd, avg_abs_inv,
           max_inv, volume, avg_spread, stress_spread, uptime, stress_uptime;
};

Summary summarize(const vector<DayResult>& rows) {
    vector<double> pnl, dd, inv, vol, sp, ssp, up, sup;
    double max_inv = 0.0;
    for (const auto& r : rows) {
        pnl.push_back(r.pnl); dd.push_back(r.max_dd); inv.push_back(r.avg_abs_inv);
        vol.push_back(r.volume); sp.push_back(r.avg_spread); ssp.push_back(r.stress_spread);
        up.push_back(r.uptime); sup.push_back(r.stress_uptime);
        max_inv = std::max(max_inv, r.max_inv);
    }
    vector<double> sorted = pnl;
    std::sort(sorted.begin(), sorted.end());
    int k = std::max(1, (int)(0.05 * sorted.size()));
    double worst = 0.0;
    for (int i = 0; i < k; i++) worst += sorted[i];
    worst /= k;

    Summary s;
    s.mean_pnl = mean_of(pnl);
    s.std_pnl = sample_std(pnl);
    s.sharpe = s.mean_pnl / s.std_pnl;
    s.cvar5 = worst;
    s.worst_day = sorted.front();
    s.max_dd = nanmean(dd);
    s.avg_abs_inv = nanmean(inv);
    s.max_inv = max_inv;
    s.volume = nanmean(vol);
    s.avg_spread = nanmean(sp);
    s.stress_spread = nanmean(ssp);
    s.uptime = nanmean(up);
    s.stress_uptime = nanmean(sup);
    return s;
}

int main(int argc, char** argv) {
    int n_days = argc > 1 ? std::atoi(argv[1]) : 120;
    unsigned long long seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2024ULL;

    MarketCfg m;
    std::mt19937_64 rng(seed);
    vector<Day> days;
    days.reserve(n_days);
    for (int i = 0; i < n_days; i++) days.push_back(generate_day(rng, m));

    vector<std::unique_ptr<Strategy>> strategies;
    strategies.emplace_back(new VanillaAS(m));
    strategies.emplace_back(new RobustAS(m, "Robust A-S (full)"));
    strategies.emplace_back(new RobustAS(m, "  - without flow predictor", false, true, true));
    strategies.emplace_back(new RobustAS(m, "  - without inventory limits", true, false, true));
    strategies.emplace_back(new RobustAS(m, "  - without jump/drawdown guard", true, true, false));

    vector<vector<DayResult>> all;
    vector<Summary> sums;
    for (auto& s : strategies) {
        vector<DayResult> rows;
        rows.reserve(n_days);
        for (const auto& d : days) rows.push_back(run_day(d, *s, m));
        sums.push_back(summarize(rows));
        all.push_back(std::move(rows));
    }

    std::printf("%d simulated days, same price/order tape for every strategy\n\n", n_days);
    std::printf("%-34s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s\n", "strategy",
                "meanPnL", "stdPnL", "Sharpe", "CVaR5%", "worst", "avgMaxDD", "avg|q|",
                "max|q|", "volume", "spread", "stressSpr", "2sideUp", "stressUp");
    for (size_t i = 0; i < strategies.size(); i++) {
        const Summary& s = sums[i];
        std::printf("%-34s%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n",
                    strategies[i]->name.c_str(), s.mean_pnl, s.std_pnl, s.sharpe, s.cvar5,
                    s.worst_day, s.max_dd, s.avg_abs_inv, s.max_inv, s.volume, s.avg_spread,
                    s.stress_spread, s.uptime, s.stress_uptime);
    }

    // paired comparison: full vs vanilla, day by day
    vector<double> d;
    for (int i = 0; i < n_days; i++) d.push_back(all[1][i].pnl - all[0][i].pnl);
    std::printf("\nPaired daily PnL diff (robust - vanilla): mean %.2f, t-stat %.2f\n",
                mean_of(d), mean_of(d) / (sample_std(d) / std::sqrt((double)d.size())));

    std::ofstream f("daily_results.csv");
    f << "strategy,day,pnl,max_dd,avg_abs_inv,max_inv,volume,avg_spread,stress_spread,uptime,stress_uptime\n";
    for (size_t s = 0; s < strategies.size(); s++) {
        std::string nm = strategies[s]->name;
        nm.erase(0, nm.find_first_not_of(' '));
        for (int i = 0; i < n_days; i++) {
            const DayResult& r = all[s][i];
            f << nm << "," << i << "," << r.pnl << "," << r.max_dd << "," << r.avg_abs_inv << ","
              << r.max_inv << "," << r.volume << "," << r.avg_spread << "," << r.stress_spread << ","
              << r.uptime << "," << r.stress_uptime << "\n";
        }
    }
    return 0;
}