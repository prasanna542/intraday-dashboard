# intraday_v4_deploy.py
"""
Intraday V4 - Deploy-ready (Full Engine Mode)
- Full 5m + 15m ensemble (same logic as your V3)
- Caching for fetch and scoring to reduce yfinance calls
- Session-state ledger (no local file writes)
- Safe UI for Streamlit Cloud (no background loops)
- Use Deep Analysis button to run heavier computations on demand
"""
from datetime import datetime, time as dtime
import math, io, time, sys
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import streamlit as st
import plotly.graph_objs as go
from functools import wraps

st.set_page_config(page_title="Intraday V4 (Deploy)", layout="wide")
MARKET_TZ = pytz.timezone('Asia/Kolkata')
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# ---------- Utility / market time ----------
def is_market_open(now=None):
    if now is None:
        now = datetime.now(MARKET_TZ)
    if now.weekday() >= 5:
        return False, 'Weekend (market closed)'
    t = now.time()
    if t < MARKET_OPEN:
        return False, f'Market not open yet (opens {MARKET_OPEN})'
    if t > MARKET_CLOSE:
        return False, f'Market already closed (closes {MARKET_CLOSE})'
    return True, 'Market open'

# ---------- Caching helpers ----------
# Simple retry wrapper for robustness
def retry(func, tries=3, delay=1.0):
    @wraps(func)
    def _inner(*args, **kwargs):
        last_exc = None
        for i in range(tries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                time.sleep(delay)
        raise last_exc
    return _inner

# Streamlit caches
@st.cache_data(ttl=60*10, show_spinner=False)  # cache results for 10 minutes
@retry
def cached_fetch_ohlcv(ticker, interval='5m', period='60d'):
    # robust fetch; raise ValueError if no data
    ticker = ticker.upper().strip()
    df = None
    # try yf.Ticker.history first
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception:
        df = None
    if df is None or df.empty:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, group_by='ticker')
        except Exception:
            df = None
    if df is None or df.empty:
        raise ValueError(f'No data returned for {ticker} (interval={interval}, period={period})')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    cols = [c.lower() if isinstance(c, str) else c for c in df.columns]
    df.columns = cols
    rename_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','adj close':'Adj Close'}
    df = df.rename(columns=rename_map)
    required = ['Open','High','Low','Close','Volume']
    if not all(r in df.columns for r in required):
        raise ValueError(f'Downloaded data for {ticker} does not contain required OHLCV. Got: {list(df.columns)}')
    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    if df.empty:
        raise ValueError(f'No valid OHLCV rows after cleaning for {ticker}')
    return df

# ---------- Indicator functions (same logic as V3) ----------
def sma(series, n): return series.rolling(n).mean()
def ema(series, n): return series.ewm(span=n, adjust=False).mean()
def atr(df, window=14):
    high = df['High']; low = df['Low']; close = df['Close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()
def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))
def macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_sig
    return macd, macd_sig, hist

def vwap(df):
    df2 = df.copy()
    df2['_date'] = df2.index.date
    out = pd.Series(index=df2.index, dtype=float)
    for d,g in df2.groupby('_date'):
        pv_series = ((g['High'] + g['Low'] + g['Close'])/3.0) * g['Volume']
        cum_pv = pv_series.cumsum()
        cum_vol = g['Volume'].cumsum().replace(0, np.nan)
        out.loc[g.index] = (cum_pv / cum_vol).fillna(method='ffill')
    return out

def vwap_slope(series, window=3):
    if len(series) < window: return 0.0
    y = series[-window:]; x = np.arange(len(y))
    if not np.all(np.isfinite(y)): return 0.0
    coef = np.polyfit(x, y, 1)[0]
    denom = float(y[-1]) if y[-1] != 0 else 1.0
    return coef/denom

def supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low'])/2
    atrv = atr(df, window=period)
    basic_ub = hl2 + multiplier * atrv
    basic_lb = hl2 - multiplier * atrv
    final_ub = basic_ub.copy(); final_lb = basic_lb.copy()
    st = pd.Series(index=df.index, dtype=float); bull = pd.Series(index=df.index, dtype=bool)
    for i in range(len(df)):
        if i == 0:
            final_ub.iloc[i] = basic_ub.iloc[i]; final_lb.iloc[i] = basic_lb.iloc[i]
            st.iloc[i] = final_ub.iloc[i]; bull.iloc[i] = True
            continue
        if basic_ub.iloc[i] < final_ub.iloc[i-1] or df['Close'].iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]
        if basic_lb.iloc[i] > final_lb.iloc[i-1] or df['Close'].iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]
        if bull.iloc[i-1]:
            if df['Close'].iloc[i] <= final_ub.iloc[i]:
                st.iloc[i] = final_ub.iloc[i]; bull.iloc[i] = False
            else:
                st.iloc[i] = final_lb.iloc[i]; bull.iloc[i] = True
        else:
            if df['Close'].iloc[i] >= final_lb.iloc[i]:
                st.iloc[i] = final_lb.iloc[i]; bull.iloc[i] = True
            else:
                st.iloc[i] = final_ub.iloc[i]; bull.iloc[i] = False
    return st, bull

def volume_spike(df, window=20, threshold=2.0):
    vol_avg = df['Volume'].rolling(window).mean()
    return df['Volume'] > (vol_avg * threshold)

# ---------- Scoring (cached) ----------
@st.cache_data(ttl=60*10, show_spinner=False)
def cached_score_df(df):
    df = df.copy()
    df['VWAP'] = vwap(df)
    df['VWAP_slope'] = df['VWAP'].rolling(3).apply(lambda s: vwap_slope(s, window=len(s)) if len(s)>=2 else 0.0)
    df['ATR14'] = atr(df, 14)
    df['RSI14'] = rsi(df['Close'], 14)
    df['MACD'], df['MACD_SIG'], df['MACD_HIST'] = macd_hist(df['Close'])
    df['VOL_SPIKE'] = volume_spike(df, window=20, threshold=2.0)
    df['ST_VAL'], df['ST_BULL'] = supertrend(df, period=10, multiplier=3)
    df['score'] = 0.0
    df['reasons'] = [[] for _ in range(len(df))]
    for i in range(5, len(df)):
        s = 0.0; reasons = []; price = df['Close'].iat[i]
        if price > df['VWAP'].iat[i]:
            s += 1.0; reasons.append('Above VWAP')
        else:
            s -= 0.6; reasons.append('Below VWAP')
        vs = float(df['VWAP_slope'].iat[i]) if not pd.isna(df['VWAP_slope'].iat[i]) else 0.0
        if vs > 0:
            s += 0.6; reasons.append('VWAP up')
        elif vs < 0:
            s -= 0.4; reasons.append('VWAP down')
        if df['ST_BULL'].iat[i]:
            s += 1.0; reasons.append('ST bull')
        else:
            s -= 1.0; reasons.append('ST bear')
        mh = df['MACD_HIST'].iat[i]
        if mh > 0:
            s += 0.7; reasons.append('MACD>0')
        else:
            s -= 0.7; reasons.append('MACD<0')
        if df['VOL_SPIKE'].iat[i]:
            s += 1.2; reasons.append('Vol spike')
        r = df['RSI14'].iat[i]
        if r > 80:
            s -= 0.9; reasons.append('RSI very high')
        if r < 20:
            s += 0.5; reasons.append('RSI very low')
        atrv = df['ATR14'].iat[i]
        if not pd.isna(atrv) and (atrv/price > 0.04):
            s -= 0.8; reasons.append('High ATR')
        df.at[df.index[i],'score'] = s
        df.at[df.index[i],'reasons'] = reasons
    df['score_norm'] = df['score'].clip(-3,3)/3.0
    df['signal'] = 0
    df.loc[df['score']>0.7,'signal'] = 1
    df.loc[df['score']<-0.7,'signal'] = -1
    return df

# ---------- Combine ----------
def combine_multi_resolution(sig5, sig15):
    last5 = sig5.iloc[-1]; last15 = sig15.iloc[-1]
    s5 = int(last5['signal']); s15 = int(last15['signal'])
    score5 = float(last5['score_norm']); score15 = float(last15['score_norm'])
    reasons = []
    reasons.extend(last5['reasons'] if isinstance(last5['reasons'], list) else [])
    reasons.extend(last15['reasons'] if isinstance(last15['reasons'], list) else [])
    if s5 == s15 and s5 != 0:
        conf = min(1.0, (abs(score5)+abs(score15))/2.0)
        action = 'BUY' if s5==1 else 'SELL'
    elif s5 != 0 and s15 == 0:
        conf = min(0.6, abs(score5)); action = 'BUY' if s5==1 else 'SELL'
    elif s15 != 0 and s5 == 0:
        conf = min(0.8, abs(score15)); action = 'BUY' if s15==1 else 'SELL'
    else:
        conf = 0.0; action = 'HOLD'
    return action, conf, reasons, {'s5':s5,'s15':s15,'score5':score5,'score15':score15}

# ---------- Sizing & SL/TP ----------
def recommend_risk_percent(balance):
    if balance < 10000: return 0.5
    if balance < 50000: return 1.0
    return 1.5
def qty_from_atr(balance, atr, risk_percent, price):
    if atr <= 0 or price <= 0: return 0
    risk_amount = balance * (risk_percent/100.0)
    stop_dist = 1.5*atr
    if stop_dist<=0: return 0
    qty = math.floor(risk_amount/stop_dist)
    return max(qty,0)
def sl_tp(price, atr, direction, rr=2.0):
    if direction == 'BUY':
        sl = price - 1.5*atr; tp = price + rr*(price - sl)
    elif direction == 'SELL':
        sl = price + 1.5*atr; tp = price - rr*(sl - price)
    else:
        return None,None
    return float(sl), float(tp)

# ---------- Session ledger (no disk writes) ----------
def init_session_ledger():
    if 'ledger' not in st.session_state:
        st.session_state.ledger = []
def log_session_ledger(ticker, interval, price, action, qty, sl, tp, conf, reasons):
    init_session_ledger()
    ts = datetime.now(MARKET_TZ).isoformat()
    st.session_state.ledger.append({'ts':ts,'ticker':ticker,'interval':interval,'price':price,'action':action,'qty':qty,'sl':sl,'tp':tp,'conf':conf,'reasons':"|".join(reasons)})

def download_ledger_csv():
    init_session_ledger()
    df = pd.DataFrame(st.session_state.ledger)
    if df.empty:
        return None
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ---------- Plotting ----------
def plot_price_with_indicators(df, title='Price'):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                 name='Price', increasing_line_color='#00B050', decreasing_line_color='#FF5555', showlegend=False))
    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(width=1)))
    if 'ST_VAL' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ST_VAL'], mode='lines', name='SuperTrend', line=dict(width=1, dash='dash')))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520)
    return fig

# ---------- UI ----------
st.title("Intraday V4 - Deploy (Full Engine)")
st.markdown("Deploy-ready full engine. Use Deep Analysis when needed. This app is cloud-optimized (caching + no background loops).")

with st.sidebar:
    st.header("Inputs")
    balance = st.number_input("Capital (INR)", value=6000.0, min_value=100.0, step=500.0)
    rec = recommend_risk_percent(balance)
    risk = st.number_input("Risk per trade %", value=float(rec), min_value=0.1, max_value=10.0, step=0.1)
    ticker = st.text_input("Ticker (Yahoo)", value="TCS.NS")
    period = st.selectbox("History period", options=["30d","60d","90d"], index=1)
    display_interval = st.selectbox("Display interval (chart)", options=["5m","15m"], index=0)
    run_fast = st.button("Quick Analysis (15m only)")
    run_deep = st.button("Deep Analysis (5m + 15m ensemble)")

st.spinner("idle")  # small placeholder

init_session_ledger()

# Quick Analysis (fast, safe - uses only 15m)
if run_fast:
    try:
        st.info("Quick analysis (15m) — fetching & scoring (cached where possible)...")
        df15 = cached_fetch_ohlcv(ticker, interval='15m', period=period)
        scored15 = cached_score_df(df15)
        last = scored15.iloc[-1]
        action = 'BUY' if last['signal']==1 else ('SELL' if last['signal']==-1 else 'HOLD')
        conf = float(last['score_norm'])
        reasons = last['reasons'] if isinstance(last['reasons'], list) else []
        price = float(last['Close']); atrv = float(last['ATR14']) if not pd.isna(last['ATR14']) else 0.0
        qty = qty_from_atr(balance, atrv, risk, price) if action in ('BUY','SELL') else 0
        sl,tp = sl_tp(price, atrv, action, rr=2.0) if action in ('BUY','SELL') else (None,None)
        is_open, status = is_market_open()
        # show results
        c1,c2,c3,c4 = st.columns([2,2,2,4])
        with c1:
            st.metric("Action", action, delta=f"conf {conf:.2f}")
            st.write("Confidence"); st.write('▮'*int(round(conf*10)) + '▯'*(10-int(round(conf*10))))
        with c2:
            st.metric("Price", f"₹{price:.2f}")
            st.write(f"Qty (suggested): {qty}")
        with c3:
            if sl is not None:
                st.metric("Stop Loss", f"₹{sl:.2f}")
                st.metric("Take Profit", f"₹{tp:.2f}")
            else:
                st.write("HOLD")
        with c4:
            st.write("Market status:"); st.write(status)
            if not is_open:
                st.warning("Market closed — historical analysis only")
        st.subheader("Reasons:")
        st.write(", ".join(reasons))
        # chart
        df_plot = scored15 if display_interval=='15m' else cached_fetch_ohlcv(ticker, interval='5m', period=period)
        if 'VWAP' not in df_plot.columns:
            df_plot['VWAP'] = vwap(df_plot)
        if 'ST_VAL' not in df_plot.columns:
            st_val, st_b = supertrend(df_plot); df_plot['ST_VAL'] = st_val
        st.plotly_chart(plot_price_with_indicators(df_plot.tail(400), title=f"{ticker} (quick)"), use_container_width=True)
        # log to session ledger
        log_session_ledger(ticker, '15m', price, action, qty, sl, tp, conf, reasons if isinstance(reasons, list) else [])
        st.success("Logged to session ledger (in-memory). Download ledger from sidebar.")
    except Exception as e:
        st.error(f"Quick analysis failed: {e}")

# Deep Analysis (full 5m + 15m ensemble)
if run_deep:
    try:
        st.info("Deep analysis — fetching 5m & 15m and running full ensemble (may take 10-30s)...")
        df5 = cached_fetch_ohlcv(ticker, interval='5m', period=period)
        df15 = cached_fetch_ohlcv(ticker, interval='15m', period=period)
        scored5 = cached_score_df(df5)
        scored15 = cached_score_df(df15)
        action, conf, reasons, meta = combine_multi_resolution(scored5, scored15)
        last5 = scored5.iloc[-1]
        price = float(last5['Close']); atrv = float(last5['ATR14']) if not pd.isna(last5['ATR14']) else 0.0
        qty = qty_from_atr(balance, atrv, risk, price) if action in ('BUY','SELL') else 0
        sl,tp = sl_tp(price, atrv, action, rr=2.0) if action in ('BUY','SELL') else (None,None)
        is_open, status = is_market_open()
        c1,c2,c3,c4 = st.columns([2,2,2,4])
        with c1:
            st.metric("Action", action, delta=f"conf {conf:.2f}")
            st.write("Confidence"); st.write('▮'*int(round(conf*10)) + '▯'*(10-int(round(conf*10))))
        with c2:
            st.metric("Price", f"₹{price:.2f}")
            st.write(f"Qty (suggested): {qty}")
        with c3:
            if sl is not None:
                st.metric("Stop Loss", f"₹{sl:.2f}")
                st.metric("Take Profit", f"₹{tp:.2f}")
            else:
                st.write("HOLD")
        with c4:
            st.write("Market status:"); st.write(status)
            if not is_open:
                st.warning("Market closed — historical analysis only")
        st.subheader("Reasons (combined):")
        st.write(", ".join(reasons))
        st.caption(f"Meta: {meta}")
        # chart - choose display interval
        df_plot = scored5 if display_interval=='5m' else scored15
        if 'VWAP' not in df_plot.columns:
            df_plot['VWAP'] = vwap(df_plot)
        if 'ST_VAL' not in df_plot.columns:
            st_val, st_b = supertrend(df_plot); df_plot['ST_VAL'] = st_val
        st.plotly_chart(plot_price_with_indicators(df_plot.tail(400), title=f"{ticker} (deep)"), use_container_width=True)
        # log
        log_session_ledger(ticker, '5m+15m', price, action, qty, sl, tp, conf, reasons if isinstance(reasons, list) else [])
        st.success("Deep analysis complete — logged to session ledger (in-memory).")
    except Exception as e:
        st.error(f"Deep analysis failed: {e}")

# Sidebar: download ledger
st.sidebar.markdown("---")
st.sidebar.header("Ledger / Export")
if st.sidebar.button("Download ledger CSV"):
    buf = download_ledger_csv()
    if buf is None:
        st.sidebar.warning("Ledger is empty.")
    else:
        st.sidebar.download_button("Download CSV", data=buf.getvalue(), file_name="session_ledger.csv", mime="text/csv")
if st.sidebar.button("Clear session ledger"):
    st.session_state.ledger = []
    st.sidebar.success("Session ledger cleared.")

st.markdown("---")
st.subheader("Session Ledger (latest entries)")
if 'ledger' in st.session_state and len(st.session_state.ledger)>0:
    df_ledger = pd.DataFrame(st.session_state.ledger)
    st.dataframe(df_ledger.tail(30))
else:
    st.write("No ledger entries yet.")

st.caption("Note: Deploy version is cloud-optimized: caching reduces repeated downloads. Use Deep Analysis sparingly to avoid API rate limits.")
