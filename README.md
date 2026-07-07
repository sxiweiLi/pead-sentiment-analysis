Earnings Sentiment Alpha — FinBERT-Driven PEAD Strategy
Does qualitative information in earnings call transcripts contain exploitable alpha beyond headline EPS numbers?
This research project answers that question with a full quantitative pipeline — from raw transcript ingestion to a backtested long-only equity strategy generating Sharpe 1.018 and +6.47% annualized alpha vs SPY.

Research Question
Post-Earnings Announcement Drift is one of the most documented anomalies in academic finance — stocks continue drifting in the direction of earnings surprises for weeks after announcement. However, raw EPS surprise signals in large-cap S&P 500 stocks have been largely arbitraged away by institutional participants.
This research asks a more specific question: can the quarter-over-quarter change in management tone — extracted via FinBERT NLP from earnings call transcripts — predict 21-day forward returns better than numerical EPS surprise filters?

Key Results
SPY buy and hold produced 13.8% CAGR. The EPS and gap-based baseline produced roughly 8-10% CAGR with Sharpe around 0.5-0.6. The sentiment long-only strategy produced 17.83% CAGR, Sharpe 1.018, Sortino 1.285, 63.2% win rate, +6.226% average return per trade, +6.47% annualized alpha vs SPY, and beta of 0.767 across 228 trades from 2018 to 2024.
The answer is yes. Sentiment velocity outperforms raw numerical EPS filters on every risk-adjusted metric.

Why This Works
The mechanism is grounded in market microstructure, not curve fitting.
At the earnings announcement, management tone improves quarter-over-quarter. From T+1 to T+5, buy-side analysts read the transcript and update their models. From T+5 to T+15, sell-side publishes upward estimate revisions. From T+15 to T+21, institutional funds rebalance toward upgraded names. At T+21, drift is complete and the position is exited.
The core insight is that EPS surprise is already partially priced by institutional consensus models before announcement. Quarter-over-quarter management tone improvement is harder to quantify — fewer participants run systematic NLP pipelines on earnings transcripts. That information asymmetry is the edge. FinBERT detects what the headline number misses: is this management team more or less confident than they were 90 days ago?

Pipeline Architecture
The project runs in eight sequential stages.
Stage 1 loads and verifies the HuggingFace dataset kurry/sp500_earnings_transcripts — 33,362 transcripts, 685 tickers, dating back to 2005.
Stage 2 filters to 20 target tickers across 2018-2024 and parses each transcript into prepared remarks and Q&A sections using speaker-turn detection. Output: 524 rows, 517 with non-empty Q&A, all 20 tickers with exactly 28 quarters.
Stage 3 preprocesses text — removes operator turns, legal boilerplate, slide references, and filler phrases. Average cleaned transcript is 2,000-5,000 words of prepared remarks and 3,400-7,000 words of Q&A.
Stage 4 runs FinBERT scoring. Each transcript section is split into 450-word chunks, scored independently, and aggregated via word-count-weighted average. Runtime is approximately 5 hours on CPU for 524 transcripts. Output is cached to parquet and never needs to be rerun.
Stage 5 engineers five features from the raw scores: prep_sentiment, qa_sentiment, sentiment_velocity (quarter-over-quarter change in prepared remarks score), sentiment_surprise (score minus ticker rolling mean), and qa_vs_prep_divergence (prepared remarks score minus Q&A score).
Stage 6 merges the transcript features with the PEAD earnings event table on ticker and nearest earnings date within a 7-day window, producing the master dataset.
Stage 7 runs Spearman IC analysis for each feature against forward returns at 5, 10, 21, 42, and 63 day horizons. Sentiment velocity is the only feature with sustained statistically significant IC, peaking at 0.108 at the 21-day horizon (t-stat approximately 2.43, p less than 0.05). All other features decay to zero or go negative by day 21.
Stage 8 backtests a long-only strategy on sentiment velocity with 21-day holds, 10% position sizing, and 10bps round-trip transaction costs.

Universe
20 large-cap S&P 500 stocks across 6 sectors. Technology: NVDA, AMD, MSFT, META, GOOGL. Consumer Discretionary: AMZN, TSLA, NKE, MCD, BKNG. Communication Services: NFLX, SPOT, DIS. Financials: JPM, GS. Healthcare: UNH, LLY. Industrials and Energy: CAT, XOM. High-beta growth: CRM.

IC Analysis Summary
Sentiment velocity IC at 5 days was +0.065, at 10 days +0.080, at 21 days +0.108 (statistically significant), at 42 days +0.069, at 63 days +0.051. All other features showed IC decaying toward zero or turning negative by the 21-day mark, indicating mean reversion rather than drift. Sentiment velocity is the sole feature with a sustained, monotonically meaningful decay curve consistent with a real drift mechanism.

Strategy Rules
Entry signal: sentiment_velocity greater than zero means long. Entry timing is the open of T+1 after the earnings date. Exit is the close of T+21 trading days from entry. Position size is 10% of current portfolio value. Maximum simultaneous positions is 10, with priority given to the highest absolute sentiment velocity when at capacity. Transaction costs are 10 basis points round trip. Starting capital is $100,000.

Full Performance Metrics
CAGR 17.83%. Sharpe ratio 1.018 at 4% risk-free rate. Sortino ratio 1.285. Calmar ratio 0.644. Max drawdown -27.69% over 507 days. Total trades 228. Win rate 63.2%. Average net return per trade +6.226%. Beta vs SPY 0.767. Annualized alpha vs SPY +6.47%.

Repository Structure
pead_analysis.ipynb is the main research notebook containing all eight stages. The data folder contains transcripts_raw.parquet, transcripts_filtered.parquet, transcripts_preprocessed.parquet, transcripts_scored.parquet, transcripts_features.parquet, and master_dataset.parquet in sequential pipeline order. The outputs folder contains ic_decay_curves.png, sentiment_velocity_equity.png, sentiment_velocity_drawdown.png, and sentiment_velocity_trades.png.

Replication
Install transformers, torch, pandas, numpy, scipy, matplotlib, seaborn, yfinance, datasets, pyarrow, and nbformat. Clone the repository and run cells sequentially in pead_analysis.ipynb. Stages 1-3 take approximately 5 minutes. Stage 4 takes approximately 5 hours on CPU — add device=0 to the FinBERT pipeline call if a GPU is available to reduce this to under 10 minutes. Stages 5-8 take approximately 10 minutes.

Honest Limitations
Survivorship bias: the universe uses current S&P 500 constituents. Companies delisted or dropped between 2018-2024 are excluded, which likely overstates long-side performance.
Bull market sample: 2018-2024 is a predominantly rising market. The short side showed only 51.4% win rate, consistent with structural difficulty of shorting large-caps on qualitative signals during sustained uptrends.
In-sample only: the full 2018-2024 period was used for both signal discovery and backtesting. Out-of-sample validation on a 2022-2024 holdout is the most critical next step before drawing strong conclusions.
Beta of 0.767: the strategy carries meaningful market exposure. True market-neutral construction would require a matched short book.

Next Steps
Near term: out-of-sample validation using 2018-2021 as training and 2022-2024 as holdout. Robustness check at 15-day and 30-day hold periods. Velocity threshold optimization using top tercile only rather than any positive reading.
Medium term: S&P 400 mid-cap universe expansion. Thinner analyst coverage in mid-caps implies slower information diffusion — expected IC in the 0.15-0.20 range. Dual-confirmation signal combining sentiment velocity with EPS surprise direction.
Longer term: real-time pipeline for live transcript ingestion and signal generation. Options overlay using sentiment velocity to time long calls around earnings rather than equity positions.

Research Context
This project sits at the intersection of two separately studied phenomena — the PEAD literature originating with Bernard and Thomas 1989, and textual analysis in finance from Tetlock 2007 and Loughran and McDonald 2011. The contribution is combining both: showing that the change in qualitative tone is more predictive than either absolute tone or raw EPS surprise at the 21-day post-earnings horizon, specifically in the modern large-cap universe where numerical PEAD has been arbitraged away.

Author
Siwei Li - aspiring systematic trader/researcher. built as part of ongoing research in equity strategies, market microstructure, and NLP-driven alpha generation. 

All results are for research purposes only. Past performance does not guarantee future results. This is not investment advice and I'm not an expert financial advisor.
