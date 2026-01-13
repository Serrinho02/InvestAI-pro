import time
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import logging
import toml
from supabase import create_client

# Importiamo la logica e la lista base
from logic import evaluate_strategy_full, POPULAR_ASSETS

# --- CONFIGURAZIONE ---
BATCH_SIZE = 300
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONNESSIONE DB ---
try:
    secrets = toml.load(".streamlit/secrets.toml")
    if "connections" in secrets and "supabase" in secrets["connections"]:
        url = secrets["connections"]["supabase"]["SUPABASE_URL"]
        key = secrets["connections"]["supabase"]["SUPABASE_KEY"]
    else:
        url = secrets.get("SUPABASE_URL")
        key = secrets.get("SUPABASE_KEY")

    if not url or not key: raise ValueError("Credenziali Supabase mancanti.")
    supabase = create_client(url, key)
    logger.info("✅ Connessione Supabase stabilita.")
except Exception as e:
    logger.error(f"❌ Configurazione: {e}")
    exit()

# --- FUNZIONI ---
def get_all_unique_tickers():
    """Recupera tutti gli asset da monitorare."""
    db_tickers = set()
    try:
        current_idx = 0
        while True:
            res = supabase.table("user_assets").select("symbol").range(current_idx, current_idx + 999).execute()
            if not res.data: break
            for row in res.data:
                if row.get('symbol'): db_tickers.add(row.get('symbol').strip().upper())
            if len(res.data) < 1000: break
            current_idx += 1000
    except Exception as e:
        logger.error(f"Errore DB: {e}")

    default_tickers = set([t.upper() for t in POPULAR_ASSETS.values() if t])
    return sorted(list(db_tickers | default_tickers))

def calculate_indicators(df):
    if len(df) < 350: return None
    df = df.copy()
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_SIGNAL'] = macd.iloc[:, 2]
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            df['BBL'] = bb.iloc[:, 0]
            df['BBU'] = bb.iloc[:, 2]
        return df
    except: return None

def process_batch(tickers):
    if not tickers: return
    try:
        data = yf.download(tickers, period="3y", group_by='ticker', progress=False, threads=True, auto_adjust=False)
    except Exception as e:
        logger.error(f"Download error: {e}")
        return

    analysis_records = []

    for ticker in tickers:
        try:
            if len(tickers) > 1:
                try:
                    if ticker in data.columns.levels[0]: df = data[ticker].copy()
                    else: continue
                except: continue
            else:
                df = data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(0)

            if df.empty or 'Close' not in df.columns: continue
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            
            # Calcolo indicatori
            df = calculate_indicators(df)
            if df is None: continue
            
            df_calc = df.dropna()
            if df_calc.empty: continue

            # Strategia
            res = evaluate_strategy_full(df_calc)
            (trend_lbl, action, color, price, rsi, dd, reason, tgt, pot, 
             risk, risk_pot, w30, p30, w60, p60, w90, p90, conf) = res

            # Prepariamo SOLO il record di analisi (molto leggero)
            analysis_records.append({
                "symbol": ticker,
                "price": float(price),
                "rsi": float(rsi),
                "trend": trend_lbl,
                "action": action,
                "confidence": int(conf),
                "target": float(tgt) if tgt else 0.0,
                "potential": float(pot),
                "risk": float(risk) if risk else 0.0,
                "risk_pot": float(risk_pot),
                "reason": reason,
                "color": color,
                "drawdown": float(dd),
                "w30": float(w30) if w30 is not None else None, 
                "p30": float(p30) if p30 is not None else None,
                "w60": float(w60) if w60 is not None else None,
                "p60": float(p60) if p60 is not None else None,
                "w90": float(w90) if w90 is not None else None,
                "p90": float(p90) if p90 is not None else None,
                "updated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.debug(f"{ticker} skipped: {e}")
            continue

    # Upsert solo di market_analysis
    if analysis_records:
        try:
            supabase.table("market_analysis").upsert(analysis_records).execute()
            logger.info(f"✅ Analizzati e salvati {len(analysis_records)} asset.")
        except Exception as e:
            logger.error(f"Errore DB Upsert: {e}")

def run_worker():
    all_tickers = get_all_unique_tickers()
    if not all_tickers: return
    logger.info(f"🚀 START WORKER - {len(all_tickers)} Asset totali")
    
    for i in range(0, len(all_tickers), BATCH_SIZE):
        process_batch(all_tickers[i:i + BATCH_SIZE])
        time.sleep(2)

if __name__ == "__main__":
    run_worker()