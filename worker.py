import time
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import logging
import toml
from supabase import create_client
import concurrent.futures # Per il parallelismo

# Importiamo la logica e la lista base
from logic import evaluate_strategy_full, POPULAR_ASSETS

# --- CONFIGURAZIONE ---
BATCH_SIZE = 300 # Scarica 300 asset alla volta (ottimo per YFinance)
MAX_WORKERS = 8  # Numero di thread paralleli per i calcoli (aumenta la velocità)

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
    """Calcola gli indicatori tecnici."""
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

def analyze_single_ticker(ticker, df):
    """
    Analizza un singolo ticker. Funzione isolata per il parallelismo.
    Restituisce un dizionario dati o None se fallisce.
    """
    try:
        # Pulizia base
        if df.empty or 'Close' not in df.columns: return None
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Calcolo Indicatori
        df = calculate_indicators(df)
        if df is None: return None
        
        # Rimozione NaN generati dagli indicatori
        df_calc = df.dropna()
        if df_calc.empty: return None

        # Strategia
        res = evaluate_strategy_full(df_calc)
        (trend_lbl, action, color, price, rsi, dd, reason, tgt, pot, 
         risk, risk_pot, w30, p30, w60, p60, w90, p90, conf) = res

        # Creazione Record
        return {
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
        }
    except Exception as e:
        # logger.debug(f"Errore analisi {ticker}: {e}")
        return None

def process_batch(tickers):
    """Scarica e processa un batch di ticker in parallelo."""
    if not tickers: return 0
    
    # 1. Download Bulk (Veloce)
    try:
        data = yf.download(tickers, period="3y", group_by='ticker', progress=False, threads=True, auto_adjust=False)
    except Exception as e:
        logger.error(f"Download error: {e}")
        return 0

    analysis_records = []
    
    # 2. Preparazione dati per il parallelismo
    tasks = []
    
    # Gestione MultiIndex di yfinance
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                # Se il ticker non è nel download, salta
                if ticker not in data.columns.levels[0]: continue
                df = data[ticker].copy()
            else:
                df = data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(0)
            
            tasks.append((ticker, df))
        except: continue

    # 3. Esecuzione Parallela (Multithreading)
    # Questo velocizza drasticamente i calcoli di pandas_ta
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Mappa la funzione analyze_single_ticker sulla lista di tasks
        # future_to_ticker = {executor.submit(analyze_single_ticker, t, df): t for t, df in tasks}
        results = executor.map(lambda x: analyze_single_ticker(x[0], x[1]), tasks)
        
        for res in results:
            if res:
                analysis_records.append(res)

    # 4. Upsert Bulk (Salva tutto in una volta)
    if analysis_records:
        try:
            # Upsert in blocchi da 100 per non sovraccaricare il payload HTTP di Supabase
            chunk_size = 100
            for i in range(0, len(analysis_records), chunk_size):
                chunk = analysis_records[i:i+chunk_size]
                supabase.table("market_analysis").upsert(chunk).execute()
            
            logger.info(f"✅ Batch salvato: {len(analysis_records)} asset.")
            return len(analysis_records)
        except Exception as e:
            logger.error(f"Errore DB Upsert: {e}")
            return 0
    return 0

def run_worker(progress_callback=None, stop_event=None):
    """Funzione principale con supporto GUI."""
    all_tickers = get_all_unique_tickers()
    total_assets = len(all_tickers)
    
    if not all_tickers: 
        if progress_callback: progress_callback(1.0, "Nessun asset trovato.")
        return

    logger.info(f"🚀 START WORKER - {total_assets} Asset")
    start_time = time.time()
    processed_count = 0
    
    for i in range(0, total_assets, BATCH_SIZE):
        # Stop manuale
        if stop_event and stop_event.is_set():
            if progress_callback: progress_callback(processed_count / total_assets, "⛔ Interrotto.")
            return

        batch_tickers = all_tickers[i:i + BATCH_SIZE]
        
        # Aggiornamento GUI
        if progress_callback:
            pct = i / total_assets
            elapsed = time.time() - start_time
            # Stima tempo
            if i > 0:
                rate = i / elapsed
                remaining = total_assets - i
                eta = remaining / rate
                eta_str = f"{int(eta//60)}m {int(eta%60)}s"
            else:
                eta_str = "Calcolo..."
            
            progress_callback(pct, f"⏳ Batch {i//BATCH_SIZE + 1} ({len(batch_tickers)} asset). ETA: {eta_str}")

        # Processamento
        processed_count += process_batch(batch_tickers)
        
        # Pausa etica per Yahoo
        time.sleep(1)

    total_time = (time.time() - start_time) / 60
    logger.info(f"🏁 FINE. Tempo: {total_time:.1f} min.")
    
    if progress_callback:
        progress_callback(1.0, f"✅ Fatto! {processed_count} asset aggiornati in {total_time:.1f} min.")

if __name__ == "__main__":
    run_worker()
