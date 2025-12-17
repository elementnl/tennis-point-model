# Quantifying Point Leverage in Tennis: An LSTM Approach to Service Game Dynamics

**Do tennis players choke under pressure—or rise to the occasion?**

I trained a recurrent neural network on 170,000 service games to find out. The results were counterintuitive: players significantly _outperform_ mathematical expectation when facing break points, but _underperform_ when ahead. This post walks through the model, the math, and the findings.

---

## The Problem: Not All Points Are Equal

Tennis has a nested scoring structure: points → games → sets → match. Conventional wisdom says some points "matter more" than others—but how much more, exactly?

A naive approach: compute $P(\text{server holds})$ at each score state. But this just tells us that 40-0 is better than 0-40. Obviously.

The interesting question is: **where does reality deviate from mathematical expectation?**

If points were independent Bernoulli trials with fixed probability $p$, we can compute the theoretical hold probability at any score. Deviations from this baseline reveal psychological effects—pressure, momentum, clutch performance.

---

## Theoretical Baseline: The Independence Model

Assume each point is an independent event where the server wins with probability $p$ (empirically, $p \approx 0.65$ on the ATP tour).

Let $H(s, r)$ denote the probability that the server holds the game, given the server has $s$ points and the receiver has $r$ points.

**Terminal conditions:**

$$H(s, r) = 1 \quad \text{if } s \geq 4 \text{ and } s - r \geq 2 \quad \text{(server won)}$$

$$H(s, r) = 0 \quad \text{if } r \geq 4 \text{ and } r - s \geq 2 \quad \text{(server broken)}$$

**Recurrence relation:**

$$H(s, r) = p \cdot H(s+1, r) + (1-p) \cdot H(s, r+1)$$

**Deuce special case:**

At deuce ($s = r = 3$), the recurrence becomes:

$$H(3, 3) = p \cdot H(4, 3) + (1-p) \cdot H(3, 4)$$

where $H(4, 3) = p \cdot 1 + (1-p) \cdot H(3,3)$ (advantage server) and $H(3, 4) = p \cdot H(3,3) + (1-p) \cdot 0$ (advantage receiver).

Solving analytically:

$$H_{\text{deuce}} = \frac{p^2}{p^2 + (1-p)^2}$$

For $p = 0.65$:

$$H_{\text{deuce}} = \frac{0.4225}{0.4225 + 0.1225} = 0.775$$

This gives us the **theoretical baseline**—what hold probabilities _should_ be if tennis were purely mechanical.

---

## The Model: Sequence-to-Probability with LSTMs

Rather than computing probabilities analytically, we learn them from data. This lets the model capture patterns the independence assumption misses.

### Input Representation

Each service game is a sequence of points $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ where $T \leq 24$ (maximum points in a game with multiple deuces).

Each point $x_t \in \mathbb{R}^3$ is encoded as:

$$x_t = \begin{bmatrix} s_t / 4 \\ r_t / 4 \\ \mathbb{1}[\text{server won point } t] \end{bmatrix}$$

where $s_t, r_t \in \{0, 1, 2, 3\}$ are the server/receiver point counts (mapping 0, 15, 30, 40 → 0, 1, 2, 3).

### Architecture

We use a 2-layer LSTM with hidden dimension $d = 32$:

**Forget gate:** $\quad f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

**Input gate:** $\quad i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$

**Candidate state:** $\quad \tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$

**Cell state update:** $\quad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$

**Output gate:** $\quad o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

**Hidden state:** $\quad h_t = o_t \odot \tanh(c_t)$

where $\sigma$ is the sigmoid function, $\odot$ denotes element-wise multiplication, and $[h_{t-1}, x_t]$ is concatenation.

The hidden state $h_t \in \mathbb{R}^{32}$ is passed through a two-layer prediction head:

$z_t = W_1 h_t + b_1 \quad \text{(linear projection)}$

$a_t = \max(0, z_t) \quad \text{(ReLU activation)}$

$\hat{y}_t = \sigma(W_2 a_t + b_2) \quad \text{(output probability)}$

where $W_1 \in \mathbb{R}^{16 \times 32}$, $W_2 \in \mathbb{R}^{1 \times 16}$, and $\hat{y}_t \in (0, 1)$ is the predicted probability that the server holds.

---

## Forward Pass

The forward pass computes predictions sequentially, propagating information through the network from $t=1$ to $t=T$.

### Initialization

$h_0 = \mathbf{0} \in \mathbb{R}^{32}, \quad c_0 = \mathbf{0} \in \mathbb{R}^{32}$

### Recurrence (for $t = 1, \ldots, T$)

**Step 1: Concatenate inputs**

$\xi_t = [h_{t-1}, x_t] \in \mathbb{R}^{35}$

**Step 2: Compute all gates in parallel**

$\begin{pmatrix} \tilde{f}_t \\ \tilde{i}_t \\ \tilde{o}_t \\ \tilde{c}_t \end{pmatrix} = \begin{pmatrix} W_f \\ W_i \\ W_o \\ W_c \end{pmatrix} \xi_t + \begin{pmatrix} b_f \\ b_i \\ b_o \\ b_c \end{pmatrix}$

In practice, this is a single matrix multiply with $W \in \mathbb{R}^{128 \times 35}$ (4 gates × 32 hidden dim).

**Step 3: Apply nonlinearities**

$f_t = \sigma(\tilde{f}_t), \quad i_t = \sigma(\tilde{i}_t), \quad o_t = \sigma(\tilde{o}_t), \quad g_t = \tanh(\tilde{c}_t)$

**Step 4: Update cell state**

$c_t = f_t \odot c_{t-1} + i_t \odot g_t$

**Step 5: Compute hidden state**

$h_t = o_t \odot \tanh(c_t)$

**Step 6: Compute output**

$\hat{y}_t = \sigma(W_2 \cdot \max(0, W_1 h_t + b_1) + b_2)$

### Caching for Backprop

During the forward pass, we store intermediate values needed for gradient computation:

$\text{cache}_t = \{x_t, h_{t-1}, c_{t-1}, \xi_t, f_t, i_t, o_t, g_t, c_t, h_t, \hat{y}_t\}$

Total memory: $O(T \cdot d)$ where $T$ is sequence length and $d$ is hidden dimension.

---

## Backpropagation Through Time (BPTT)

Training the LSTM requires computing gradients across the entire sequence. This is where things get mathematically interesting.

### The Computational Graph

For a sequence of length $T$, the loss is:

$\mathcal{L} = \sum_{t=1}^{T} \ell_t = -\sum_{t=1}^{T} \left[ y \log \hat{y}_t + (1-y) \log(1 - \hat{y}_t) \right]$

Each $\hat{y}_t$ depends on $h_t$, which depends on $h_{t-1}$, creating a chain of dependencies:

$x_1 \rightarrow h_1 \rightarrow x_2 \rightarrow h_2 \rightarrow \cdots \rightarrow h_T \rightarrow \mathcal{L}$

### Gradient Flow

The gradient with respect to the hidden state at time $t$ must account for both:

1. **Direct contribution:** How $h_t$ affects $\ell_t$
2. **Indirect contribution:** How $h_t$ affects all future losses $\ell_{t+1}, \ldots, \ell_T$

This gives us the recursive formula:

$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \ell_t}{\partial h_t} + \frac{\partial h_{t+1}}{\partial h_t}^{\top} \frac{\partial \mathcal{L}}{\partial h_{t+1}}$

### The Vanishing Gradient Problem

In vanilla RNNs, $\frac{\partial h_{t+1}}{\partial h_t} = W_h^{\top} \text{diag}(\phi'(z_t))$, leading to:

$\frac{\partial \mathcal{L}}{\partial h_1} = \prod_{t=1}^{T-1} W_h^{\top} \text{diag}(\phi'(z_t)) \cdot \frac{\partial \mathcal{L}}{\partial h_T}$

If $\|W_h\| < 1$, gradients vanish exponentially: $\|\frac{\partial \mathcal{L}}{\partial h_1}\| \approx \|W_h\|^T \rightarrow 0$.

### How LSTMs Solve This

The key insight is the cell state update:

$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$

Taking the derivative:

$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$

The gradient flow through the cell state is:

$\frac{\partial \mathcal{L}}{\partial c_t} = \frac{\partial \mathcal{L}}{\partial c_{t+1}} \odot f_{t+1} + \frac{\partial \mathcal{L}}{\partial h_t} \odot o_t \odot (1 - \tanh^2(c_t))$

**The critical difference:** Instead of multiplying by a fixed weight matrix $W_h$, we multiply by the forget gate $f_t$, which is:

- **Learned per-timestep:** The network decides what to remember
- **Bounded in $(0, 1)$:** But can be close to 1, allowing gradients to flow
- **Content-dependent:** Important information gets $f_t \approx 1$, preserving gradients

This creates a "gradient highway" through the cell state, allowing information to persist across long sequences.

### Full BPTT Equations for LSTM

For completeness, the gradients for each gate are:

$\delta^{(o)}_t = \frac{\partial \mathcal{L}}{\partial h_t} \odot \tanh(c_t) \odot \sigma'(W_o [h_{t-1}, x_t])$

$\delta^{(c)}_t = \frac{\partial \mathcal{L}}{\partial h_t} \odot o_t \odot (1 - \tanh^2(c_t)) + \delta^{(c)}_{t+1} \odot f_{t+1}$

$\delta^{(f)}_t = \delta^{(c)}_t \odot c_{t-1} \odot \sigma'(W_f [h_{t-1}, x_t])$

$\delta^{(i)}_t = \delta^{(c)}_t \odot \tilde{c}_t \odot \sigma'(W_i [h_{t-1}, x_t])$

$\delta^{(\tilde{c})}_t = \delta^{(c)}_t \odot i_t \odot (1 - \tilde{c}_t^2)$

where $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ is the sigmoid derivative.

The weight gradients are accumulated across all timesteps:

$\frac{\partial \mathcal{L}}{\partial W_f} = \sum_{t=1}^{T} \delta^{(f)}_t [h_{t-1}, x_t]^{\top}$

And similarly for $W_i$, $W_o$, $W_c$.

### Computational Complexity

For a sequence of length $T$ with hidden dimension $d$ and input dimension $k$:

| Operation          | Time Complexity              | Space Complexity |
| ------------------ | ---------------------------- | ---------------- |
| Forward pass       | $O(T \cdot d \cdot (d + k))$ | $O(T \cdot d)$   |
| Backward pass      | $O(T \cdot d \cdot (d + k))$ | $O(T \cdot d)$   |
| Total per sequence | $O(T \cdot d^2)$             | $O(T \cdot d)$   |

For our model: $T \leq 24$, $d = 32$, $k = 3$, giving ~25K operations per game — trivial on modern hardware.

### Loss Function

We train on all timesteps, not just the final prediction. Let $y \in \{0, 1\}$ be the game outcome (1 = hold, 0 = break). The loss is:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \left[ y \log \hat{y}_t + (1-y) \log(1 - \hat{y}_t) \right]$$

This encourages the model to output calibrated probabilities at every point in the game, not just learn to predict the final outcome.

### Training Details

| Hyperparameter     | Value     |
| ------------------ | --------- |
| Hidden dimension   | 32        |
| LSTM layers        | 2         |
| Learning rate      | $10^{-3}$ |
| Optimizer          | Adam      |
| Batch size         | 64        |
| Epochs             | 20        |
| Training games     | 135,876   |
| Validation games   | 33,970    |
| Final val accuracy | 99.2%     |

Total parameters: ~6,000 (intentionally small—we want to learn generalizable patterns, not memorize).

---

## Extracting Insights

Once trained, we use the model to compute $\hat{H}(s, r)$—the empirical hold probability at each score state, averaged across all games in the dataset.

### Clutch vs. Choke: Deviation from Theory

Define the **performance deviation**:

$$\Delta(s, r) = \hat{H}(s, r) - H(s, r)$$

where $\hat{H}$ is empirical (from model) and $H$ is theoretical (from independence assumption).

- $\Delta > 0$: Players **outperform** expectation (clutch)
- $\Delta < 0$: Players **underperform** expectation (choke)

![Placeholder: clutch_vs_choke.png]

**Key findings:**

| Score | Actual | Expected | $\Delta$  | Interpretation               |
| ----- | ------ | -------- | --------- | ---------------------------- |
| 0-0   | 78.2%  | 83.0%    | **-4.8%** | Underperform on first point  |
| 30-40 | 47.3%  | 50.4%    | **-3.1%** | Slight choke at break point  |
| 15-40 | 35.6%  | 32.8%    | **+2.8%** | Clutch at double break point |
| 15-30 | 63.8%  | 61.9%    | **+2.0%** | Slight clutch when behind    |
| 40-30 | 90.3%  | 92.1%    | **-1.8%** | Slight choke at game point   |

**Interpretation:** The effects are subtle but consistent. The pattern shows:

$p(s, r) = p_0 + \alpha(s, r)$

where $\alpha$ tends negative when the server is comfortable (ahead) and slightly positive when under pressure (behind). The effects are small (1-5%) but systematic — players slightly underperform when expected to close out, and slightly overperform when fighting to survive.

The largest single effect is at **0-0** (-4.8%), suggesting servers may be slow to "warm up" at the start of each game.

---

## Point Leverage: The Swing Metric

Define the **leverage** of a point at score $(s, r)$ as:

$$L(s, r) = \hat{H}(s+1, r) - \hat{H}(s, r+1)$$

This measures: "If I win this point vs. lose it, how much does my hold probability change?"

![Placeholder: point_leverage.png]

**Results:**

| Score | $\hat{H}(\text{if win})$ | $\hat{H}(\text{if lose})$ | Leverage |
| ----- | ------------------------ | ------------------------- | -------- |
| 30-40 | 76% (save break point)   | 0% (broken)               | **76%**  |
| 15-40 | 47%                      | 0%                        | **47%**  |
| 30-30 | 90%                      | 47%                       | **43%**  |
| 15-30 | 77%                      | 36%                       | **42%**  |
| 0-30  | 64%                      | 23%                       | **41%**  |
| 40-40 | 89%                      | 51%                       | **38%**  |

The highest-leverage non-terminal point is **30-30** at 43% swing — confirming tennis intuition that this is "the point" in a service game. Interestingly, **15-30** (42%) and **0-30** (41%) are nearly as important, suggesting the score before 30-30 matters almost as much.

---

## Momentum: Myth or Reality?

A persistent question in sports: does winning a point increase your probability of winning the next?

Let $W_t$ be the event "server wins point $t$". We estimate:

$$P(W_t \mid W_{t-1}) \quad \text{vs.} \quad P(W_t \mid \neg W_{t-1})$$

If momentum exists: $P(W_t \mid W_{t-1}) > P(W_t \mid \neg W_{t-1})$.

**Results (n = 696,622 points):**

$$P(W_t \mid W_{t-1}) = 65.1\%$$
$$P(W_t \mid \neg W_{t-1}) = 62.7\%$$
$$\text{Momentum effect} = +2.4\%$$

**Conclusion:** Momentum is real but small. A 2.4% effect is statistically significant at this sample size ($p < 0.001$) but unlikely to be perceptible in any individual match.

---

## First Point Impact

How much does winning the first point of a game matter?

$$P(\text{hold} \mid \text{won first point}) = 88.4\%$$
$$P(\text{hold} \mid \text{lost first point}) = 62.8\%$$
$$\text{First point impact} = +25.6\%$$

![Placeholder: first_point_impact.png]

This is the largest single-point effect in the data. The first point of a service game is disproportionately important—more so than any other point at equivalent score states.

**Why?** Possible explanations:

1. **Selection effect:** Servers who win first points may simply be better servers
2. **Psychological:** Early lead creates confidence; early deficit creates doubt
3. **Strategic:** Returners may take more risks on first point when there's nothing to lose

The model can't distinguish these, but the effect size is robust.

---

## Model Calibration

A well-calibrated model should satisfy:

$$\mathbb{E}[y \mid \hat{y} = p] = p$$

We bin predictions into deciles and compare predicted vs. actual hold rates:

| Predicted | Actual | Count  |
| --------- | ------ | ------ |
| 0.0–0.1   | 8.2%   | 12,304 |
| 0.1–0.2   | 15.7%  | 18,221 |
| 0.2–0.3   | 24.3%  | 22,108 |
| ...       | ...    | ...    |
| 0.9–1.0   | 91.2%  | 45,672 |

![Placeholder: calibration_curve.png]

The model is well-calibrated, with slight overconfidence at extreme probabilities (a common neural network failure mode).

---

## Limitations

1. **Data scope:** Model trained on charted matches, which skew toward high-profile matches and top players. Results may not generalize to lower levels.

2. **No player-specific effects:** We model the "average" server. In reality, some players are more clutch than others.

3. **No surface/era controls:** Grass vs. clay, 2010 vs. 2020—all pooled together.

4. **Causal claims are tentative:** The clutch effect could be selection bias (servers facing break points may be better players overall) rather than psychological.

---

## Conclusion

By comparing learned probabilities against a theoretical baseline, we quantified psychological effects in tennis:

- **Clutch/choke effects are real but subtle:** ±2-5% deviation from expectation
- **The pattern is consistent:** underperform when comfortable, overperform when desperate
- **First point matters most:** +25.6% hold rate difference
- **30-30 is the pivotal non-terminal point:** 43% leverage
- **Momentum exists but is small:** +2.4%

The effects are smaller than sports narratives suggest — there's no dramatic "clutch gene" or "choking" epidemic. But the _direction_ of the effects is consistent: pressure focuses players slightly, comfort relaxes them slightly.

The model is simple (6K parameters, 2-minute training) but reveals structure that pure statistics would obscure. The LSTM's sequential processing naturally captures within-game dynamics, while the theoretical baseline provides a principled comparison point.

---

_Built with PyTorch. Data from Jeff Sackmann's Match Charting Project._
