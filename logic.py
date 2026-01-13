import streamlit as st
from st_supabase_connection import SupabaseConnection
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import logging
from passlib.context import CryptContext
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURAZIONE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE HASHING ---
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"], 
    deprecated="auto",
    pbkdf2_sha256__default_rounds=300000 
)

# --- CONFIGURAZIONE TELEGRAM ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")

# --- ASSET LIST COMPLETA (Ottimizzata e ordinata) ---
POPULAR_ASSETS = {
    # --- INDICI GLOBALI ---
    "S&P 500 (USA)": "SPY", "Nasdaq 100 (Tech)": "QQQ", 
    "Russell 2000 (Small Cap)": "IWM", "Dow Jones": "DIA",
    "All-World": "VWCE.DE", "Emerging Markets": "EEM", "Europe Stoxx 50": "FEZ",
    "China (Large Cap)": "FXI", "China (Internet)": "KWEB", "India": "INDA",
    "Brazil": "4BRZ.DE", "Japan": "EWJ", "UK (FTSE 100)": "EWU", "Germany (DAX)": "EWG",
    # --- MATERIE PRIME & METALLI ---
    "Gold": "GLD", "Silver": "SLV", "Oil (WTI)": "USO", 
    "Natural Gas": "UNG", "Copper (Miners)": "COPX", 
    "Uranium": "URA", "Agriculture": "DBA",
    # --- IMMOBILIARE & OBBLIGAZIONI ---
    "Real Estate (US)": "VNQ", "US Treasury 20Y+": "TLT", 
    "US Treasury 1-3Y": "SHY", "Corporate Bonds": "LQD",
    # --- SETTORI USA & MEGATREND ---
    "Semiconductors": "SMH", "Technology": "XLK", "Healthcare": "XLV", 
    "Financials": "XLF", "Energy": "XLE", "Materials": "XLB",
    "Industrials": "XLI", "Consumer Disc. (Amazon/Tesla)": "XLY", 
    "Consumer Staples (Coca/Pepsi)": "XLP", "Utilities": "XLU",
    "Clean Energy": "ICLN", "Cybersecurity": "CIBR.MI", "Robotics & AI": "BOTZ",
    "Defense & Aerospace": "ITA", "Biotech": "XBI", "Tonies SE": "TNIE.F", "DroneShield": "DRH.F",
    "Redcare Pharmacy": "0RJT.IL",
    # --- CRYPTO ---
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
    "Ripple": "XRP-USD", "Binance Coin": "BNB-USD", "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD", "Chainlink": "LINK-USD", "Polkadot": "DOT-USD",
    # --- BIG TECH (USA) ---
    "Nvidia": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT", "Tesla": "TSLA", 
    "Amazon": "AMZN", "Meta": "META", "Google": "GOOGL", "Oracle": "ORCL",
    "Netflix": "NFLX", "AMD": "AMD.F", "Palantir": "PLTR", "Coinbase": "COIN",
    "monday.com": "MNDY", "Arista Networks": "ANET", "Duolingo": "DUOL", 
    "The Trade Desk": "TTD", "Axon Enterprise": "AXON", "BigBear.ai": "BBAI",
    "Installed Building Products": "IBP",
    # --- BIG EUROPE (GRANOLAS) ---
    "ASML (Chip)": "ASML", "LVMH (Luxury)": "MC.PA", 
    "Novo Nordisk (Pharma)": "NVO", "SAP (Software)": "SAP", "Easyjet": "EJT1.F",
    # --- ITALIA (FTSE MIB) ---
    "Ferrari": "RACE.MI", "Intesa Sanpaolo": "ISP.MI", "UniCredit": "UCG.MI", 
    "Enel": "ENEL.MI", "Eni": "ENI.MI", "Stellantis": "STLAM.MI", 
    "Leonardo": "LDO.MI", "Generali": "G.MI", "Moncler": "MONC.MI", 
    "Poste Italiane": "PST.MI", "Terna": "TRN.MI", "Snam": "SRG.MI", 
    "Mediobanca": "MB.MI", "Tenaris": "TEN.MI", "Prysmian": "PRY.MI",
    "Fincantieri": "FCT.MI", "Juventus": "JUV.F", "Banca Monte dei Paschi di Siena": "BMPSM.XD",
    "Banca Popolare di Sondrio": "BPSOM.XD"
} 
AUTO_SCAN_TICKERS = [v for k, v in POPULAR_ASSETS.items() if v is not None]

# --- DATABASE MANAGER (Supabase) ---
class DBManager:
    def __init__(self):
        self.db_url = "SUPABASE_API_CONNECTION_ACTIVE"
        
        try:
            self.conn = st.connection("supabase", type=SupabaseConnection)
            self.client = self.conn.client 
        except Exception as e:
            st.error(f"❌ Errore connessione Supabase: {e}")
            st.stop()

    # --- METODI UTENTE ---
    
    def register_user(self, u, p):
        h = hash_password(p)
        try:
            res = self.client.table("users").insert({"username": u, "password": h}).execute()
            return len(res.data) > 0
        except Exception as e:
            logger.error(f"Errore registrazione utente {u}: {e}")
            return False

    def login_user(self, u, p):
        try:
            res = self.client.table("users").select("password").eq("username", u).execute()
            if res.data and len(res.data) > 0:
                hashed_password = res.data[0]['password']
                return verify_password(p, hashed_password)
            return False
        except Exception as e:
            logger.error(f"Errore login utente {u}: {e}")
            return False

    def change_password(self, username, new_password):
        h = hash_password(new_password)
        try:
            res = self.client.table("users").update({"password": h}).eq("username", username).execute()
            return len(res.data) > 0
        except Exception as e:
            logger.error(f"Errore cambio password: {e}")
            return False
    
    def save_chat_id(self, user, chat_id):
        try:
            res = self.client.table("users").update({"tg_chat_id": str(chat_id)}).eq("username", user).execute()
            return len(res.data) > 0
        except Exception as e:
            logger.error(f"Errore save_chat_id: {e}")
            return False

    def disable_notifications(self, user, chat_id):
        try:
            stop_id = f"STOP_{chat_id}"
            res = self.client.table("users").update({"tg_chat_id": stop_id}).eq("username", user).execute()
            return len(res.data) > 0
        except Exception as e:
            logger.error(f"Errore disable_notifications: {e}")
            return False  

    def get_user_chat_id(self, user):
        try:
            res = self.client.table("users").select("tg_chat_id").eq("username", user).execute()
            if res.data and len(res.data) > 0:
                raw_id = res.data[0].get("tg_chat_id", "")
                return raw_id.replace("STOP_", "")
            return ""
        except Exception as e:
            logger.error(f"Errore get_user_chat_id per {user}: {e}")
            return ""

    def get_users_with_telegram(self):
        try:
            res = self.client.table("users").select("username, tg_chat_id").neq("tg_chat_id", "").execute()
            return [(r['username'], r['tg_chat_id']) for r in res.data if not r['tg_chat_id'].startswith("STOP_")]
        except Exception as e:
            logger.error(f"Errore get_users_with_telegram: {e}")
            return []

    def get_user_by_chat_id(self, chat_id):
        try:
            res = self.client.table("users").select("username").eq("tg_chat_id", str(chat_id)).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]['username']
            stop_id = f"STOP_{chat_id}"
            res = self.client.table("users").select("username").eq("tg_chat_id", stop_id).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]['username']
            return None
        except Exception as e:
            logger.error(f"Errore get_user_by_chat_id: {e}")
            return None

    # --- METODI TRANSAZIONI ---
    
    def add_transaction(self, user, symbol, qty, price, date_str, type="BUY", fee=0.0):
        data = {
            "username": user,
            "symbol": symbol.upper(),
            "quantity": float(qty),
            "price": float(price),
            "date": date_str,
            "type": type,
            "fee": float(fee)
        }
        try:
            self.client.table("transactions").insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Errore aggiunta transazione per {user} ({symbol}): {e}")
            return False

    def update_transaction(self, t_id, symbol, qty, price, date_str, type, fee=0.0):
        data = {
            "symbol": symbol.upper(),
            "quantity": float(qty),
            "price": float(price),
            "date": date_str,
            "type": type,
            "fee": float(fee)
        }
        try:
            self.client.table("transactions").update(data).eq("id", t_id).execute()
            return True
        except Exception as e:
            logger.error(f"Errore aggiornamento transazione ID {t_id}: {e}")
            return False

    def delete_transaction(self, t_id):
        try:
            self.client.table("transactions").delete().eq("id", t_id).execute()
            return True
        except Exception as e:
            logger.error(f"Errore eliminazione transazione ID {t_id}: {e}")
            return False

    def get_all_transactions(self, user):
        try:
            res = self.client.table("transactions").select("*").eq("username", user).order("date", desc=True).execute()
            
            clean_rows = []
            for r in res.data:
                clean_rows.append((
                    r['id'], 
                    r['symbol'], 
                    float(r['quantity']), 
                    float(r['price']), 
                    r['date'], 
                    r['type'], 
                    float(r.get('fee', 0.0))
                ))
            return clean_rows
        except Exception as e:
            logger.error(f"Errore fetch tx: {e}")
            return []

    def get_transaction_by_id(self, t_id):
        try:
            res = self.client.table("transactions").select("*").eq("id", t_id).execute()
            if res.data:
                r = res.data[0]
                return (r['id'], r['symbol'], r['quantity'], r['price'], r['date'], r['type'], r.get('fee', 0.0))
            return None
        except Exception as e:
            logger.error(f"Errore get_transaction_by_id per ID {t_id}: {e}")
            return None

    def get_portfolio_summary(self, user):
        rows = self.get_all_transactions(user)
        portfolio = {}
        history = [] 
        
        for row in rows:
            sym = row[1]
            qty = float(row[2])
            price = float(row[3])
            dt = row[4]
            type_tx = row[5]
            fee = float(row[6])

            if sym not in portfolio: 
                portfolio[sym] = {"qty": 0.0, "total_cost": 0.0, "avg_price": 0.0}
            
            if type_tx == "BUY":
                portfolio[sym]["qty"] += qty
                portfolio[sym]["total_cost"] += (qty * price) + fee
            elif type_tx == "SELL":
                portfolio[sym]["qty"] -= qty
                portfolio[sym]["total_cost"] -= (qty * price)
            
            if portfolio[sym]["qty"] > 0.000001:
                portfolio[sym]["avg_price"] = portfolio[sym]["total_cost"] / portfolio[sym]["qty"]
            else:
                portfolio[sym]["avg_price"] = 0
                portfolio[sym]["total_cost"] = 0
                
            history.append({"symbol": sym, "date": dt, "price": price, "type": type_tx, "fee": fee})

        return {k: v for k, v in portfolio.items() if v["qty"] > 0.0001}, history


# --- METODI READ-ONLY (Dati di Mercato da DB) ---
    
    def get_analysis_snapshot(self):
        """
        Recupera l'intera tabella market_analysis per lo scanner.
        Molto più veloce che ricalcolare tutto.
        """
        try:
            # Prende tutto. Se hai 10k asset, potrebbe servire paginazione o filtri, 
            # ma per ora 10k righe in JSON sono gestibili (~2-3MB)
            res = self.client.table("market_analysis").select("*").execute()
            
            # Convertiamo in una lista di dizionari facile da consumare per l'UI
            return res.data
        except Exception as e:
            logger.error(f"Errore get_analysis_snapshot: {e}")
            return []

    def get_market_data_for_asset(self, symbol, days=365):
        """
        Recupera storico OHLCV + Indicatori dal DB per i grafici.
        """
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            res = self.client.table("market_data")\
                .select("*")\
                .eq("symbol", symbol)\
                .gte("date", cutoff)\
                .order("date", desc=False)\
                .execute()
            
            if not res.data:
                return pd.DataFrame()
                
            df = pd.DataFrame(res.data)
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            
            # Rinomina colonne per compatibilità con il resto del codice (Capital case per grafici Plotly)
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                'sma_50': 'SMA_50', 'sma_200': 'SMA_200', 'rsi': 'RSI'
            }, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Errore get_market_data_for_asset {symbol}: {e}")
            return pd.DataFrame()

# --- METODI LISTA PREFERITI (Watchlist) ---
    
    def get_user_watchlist(self, user):
        """Recupera la lista asset personalizzata dell'utente."""
        try:
            res = self.client.table("user_assets").select("*").eq("username", user).execute()
            # Ritorna un dizionario {Nome: Symbol} compatibile con la UI
            watchlist = {row['name'] if row['name'] else row['symbol']: row['symbol'] for row in res.data}
            return watchlist
        except Exception as e:
            logger.error(f"Errore get_watchlist per {user}: {e}")
            return {}

    def add_asset_to_watchlist(self, user, symbol, name=None):
        """Aggiunge un asset alla lista personale."""
        if not name: name = symbol
        try:
            self.client.table("user_assets").insert({
                "username": user, 
                "symbol": symbol.upper(),
                "name": name
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Errore add_asset: {e}")
            return False

    def remove_asset_from_watchlist(self, user, symbol):
        """Rimuove un asset."""
        try:
            self.client.table("user_assets").delete().eq("username", user).eq("symbol", symbol).execute()
            return True
        except Exception as e:
            logger.error(f"Errore remove_asset: {e}")
            return False
            
    def init_default_watchlist(self, user):
        """Se l'utente è nuovo, copia i POPULAR_ASSETS nella sua lista."""
        # Se ha già asset, non fare nulla
        existing = self.get_user_watchlist(user)
        if existing: return
        
        # Altrimenti popola con i default
        # Nota: Importa POPULAR_ASSETS localmente per evitare cicli se necessario, 
        # oppure passalo come argomento. Qui assumiamo sia disponibile globalmente in logic.py
        for name, sym in POPULAR_ASSETS.items():
            self.add_asset_to_watchlist(user, sym, name)


# --- HELPER HASHING ---
def hash_password(password) -> str:
    p_str = str(password) if password else ""
    if not p_str:
        p_str = ""
    return pwd_context.hash(p_str)

def verify_password(plain_password, hashed_password: str) -> bool:
    p_str = str(plain_password) if plain_password else ""
    return pwd_context.verify(p_str, hashed_password)

# --- HELPER FUNCTIONS ---
def validate_ticker(ticker):
    if not ticker: return False
    try:
        t = yf.Ticker(ticker)
        return len(t.history(period="1d")) > 0
    except Exception as e:
        logger.debug(f"Validazione fallita per {ticker}: {e}")
        return False

def get_data_raw(tickers):
    """
    Download universale con gestione MultiIndex migliorata.
    """
    if not tickers: return {}
    data = {}
    
    unique_tickers = list(set([t.strip().upper() for t in tickers if t]))
    if not unique_tickers: return {}

    try:
        df = yf.download(unique_tickers, period="3y", group_by='ticker', progress=False, auto_adjust=False)
        
        if df.empty:
            return {}

        for t in unique_tickers:
            asset_df = pd.DataFrame()
            
            try:
                if isinstance(df.columns, pd.MultiIndex) and t in df.columns.get_level_values(0):
                    asset_df = df[t].copy()
                elif len(unique_tickers) == 1:
                    if 'Close' in df.columns:
                        asset_df = df.copy()
                    elif isinstance(df.columns, pd.MultiIndex):
                        asset_df = df.copy()
                        asset_df.columns = asset_df.columns.get_level_values(0)
            
                if not asset_df.empty and 'Close' in asset_df.columns:
                    asset_df.dropna(subset=['Close'], inplace=True)
                    process_df(asset_df, data, t)
                    
            except Exception as e:
                logger.debug(f"Errore estrazione {t}: {e}")
                continue
                
        return data

    except Exception as e:
        logger.error(f"Errore download generale: {e}")
        return {}

def process_df(df, data, t):
    df = df.dropna(how='all')
    if len(df_clean) < 300:
        return
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        macd = ta.macd(df['Close'])
        if macd is not None and not macd.empty:
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_SIGNAL'] = macd.iloc[:, 2]
        else:
            return   
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None and not bb.empty:
            df['BBL'] = bb.iloc[:, 0]
            df['BBU'] = bb.iloc[:, 2]
        else:
            return    
        df_clean = df.dropna()
        if not df_clean.empty:
            data[t] = df_clean     
    except Exception as e:
        logger.warning(f"Errore calcolo indicatori per {t}: {e}")

# --- BACKTESTING E PROBABILITÀ ---
def run_backtest(df, days_list=[30, 60, 90]):
    df = df.copy()
    df['Signal'] = (
        (df['Close'] > df['SMA_200']) &
        ((df['RSI'] < 40) | (df['Close'] <= df['BBL'] * 1.01))
    )
    df['Signal'] = df['Signal'].shift(1)
    entry_indices = df.index[df['Signal'] == True].tolist()
    results = {
        d: {"wins": 0, "pnl_sum": 0.0, "valid": 0}
        for d in days_list
    }
    for entry_date in entry_indices:
        idx = df.index.get_loc(entry_date)
        entry_price = df.iloc[idx]['Close']
        for d in days_list:
            if idx + d < len(df):
                exit_price = df.iloc[idx + d]['Close']
                pnl = (exit_price - entry_price) / entry_price
                results[d]["valid"] += 1
                results[d]["pnl_sum"] += pnl
                if pnl > 0:
                    results[d]["wins"] += 1
    out = []
    for d in days_list:
        if results[d]["valid"] > 0:
            win = results[d]["wins"] / results[d]["valid"] * 100
            pnl = results[d]["pnl_sum"] / results[d]["valid"] * 100
            out.extend([round(win, 1), round(pnl, 2)])
        else:
            out.extend([None, None])
    return tuple(out)


# --- CONFIDENCE SCORE MIGLIORATO ---
def calculate_confidence(df, is_bullish, action, potential_upside, potential_downside, w30, p30, w60, p60, w90, p90):
    """
    Score 0-100 con pesatura ottimizzata.
    """
    last_close = df['Close'].iloc[-1]
    last_sma50 = df['SMA_50'].iloc[-1]
    last_sma200 = df['SMA_200'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    
    # 1. Trend Strength (30%)
    trend_score = 0
    if is_bullish:
        trend_factor = min(1.0, (last_close - last_sma200) / last_sma200 * 10)
        if last_sma50 > last_sma200:
             trend_factor *= 1.2
        trend_score = min(30, trend_factor * 30)

    # 2. Setup Quality (25%)
    setup_score = 0
    if "ORO" in action:
        setup_score = 25
    else:
        setup_factor = max(0, 40 - last_rsi) / 20 
        setup_score = min(25, setup_factor * 25)

    # 3. Risk/Reward (15%)
    risk_score = 0
    if potential_upside > 0 and potential_downside < 0:
        risk_reward_ratio = potential_upside / abs(potential_downside) 
        risk_score = min(15, (risk_reward_ratio / 2) * 15)

    # 4. Backtest (30%)
    backtest_score = 0
    if w30 and w60 and w90:
        win_rate_weight = ((w30 + w60) / 200) * 15
        pnl_90_weight = max(0, min(15, (p90 or 0) / 3))
        backtest_score = min(30, win_rate_weight + pnl_90_weight)

    confidence = trend_score + setup_score + risk_score + backtest_score
    return round(min(100, max(0, confidence)))

# --- STRATEGIA MIGLIORATA ---
def evaluate_strategy_full(df):
    """
    Strategia con gestione errori potenziata.
    """
    required_cols = ['SMA_200', 'MACD', 'MACD_SIGNAL', 'BBL', 'BBU', 'RSI', 'ATR']
    for col in required_cols:
        if col not in df.columns:
            return "N/A", "Dati insufficienti", "#eee", 0, 50, 0, "Mancano indicatori", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    if df.empty: 
        return "N/A", "Dati insufficienti", "#eee", 0, 50, 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    try:
        last_close = df['Close'].iloc[-1]
        last_sma200 = df['SMA_200'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        last_macd = df['MACD'].iloc[-1]
        last_macd_signal = df['MACD_SIGNAL'].iloc[-1]
        last_bbl = df['BBL'].iloc[-1]
        last_bbu = df['BBU'].iloc[-1]
        last_atr = df['ATR'].iloc[-1]
        
        max_price = df['Close'].max()
        drawdown = ((last_close - max_price) / max_price) * 100
        
        is_bullish = last_close > last_sma200
        trend_label = "BULLISH (Rialzista)" if is_bullish else "BEARISH (Ribassista)"
        
        action = "✋ ATTENDI"
        color = "#fcfcfc"
        reason = "Nessun segnale operativo chiaro."
        
        technical_target = max(last_bbu, last_close + (2 * last_atr))
        potential_upside = ((technical_target - last_close) / last_close) * 100

        technical_risk = min(last_bbl, last_close - (2 * last_atr))
        potential_downside = ((technical_risk - last_close) / last_close) * 100

        try:
            w30, p30, w60, p60, w90, p90 = run_backtest(df) 
            w30 = w30 or 0
            p30 = p30 or 0
            w60 = w60 or 0
            p60 = p60 or 0
            w90 = w90 or 0
            p90 = p90 or 0
        except Exception as e:
            logger.error(f"Errore backtest: {e}")
            w30, p30, w60, p60, w90, p90 = 0, 0, 0, 0, 0, 0

        if is_bullish:
            if last_rsi < 30 and last_close <= last_bbl:
                action = "💎 OPPORTUNITÀ D'ORO"
                color = "#FFD700"
                reason = "RARITÀ: Asset in trend rialzista crollato a livelli estremi."
            elif last_rsi < 40 or last_close <= last_bbl * 1.01: 
                action = "🛒 ACQUISTA ORA! (Dip)"
                color = "#ccffcc"
                reason = "Trend rialzista + Prezzo a sconto."
            elif last_rsi > 75 or (last_close >= last_bbu and last_macd < last_macd_signal):
                 action = "💰 VENDI PARZIALE"
                 color = "#ffdddd"
                 reason = "Prezzo esteso. Rischio ritracciamento."
                 technical_risk = last_sma200 
                 potential_downside = ((technical_risk - last_close) / last_close) * 100
            else:
                 action = "🚀 TREND SOLIDO"
                 color = "#e6f4ea"
                 reason = "Il trend è sano. Lascia correre."
        else:
            if last_rsi < 30 and last_close < last_bbl:
                 action = "⚠️ TENTATIVO RISCHIOSO"
                 color = "#fff4cc"
                 reason = "Rimbalzo tecnico (Dead Cat Bounce)."
            elif last_macd < last_macd_signal:
                 action = "⛔ STAI ALLA LARGA"
                 color = "#fcfcfc"
                 reason = "Trend ribassista. Momentum negativo."
                 potential_upside = 0

        confidence_score = calculate_confidence(
            df, is_bullish, action, potential_upside, potential_downside, 
            w30, p30, w60, p60, w90, p90
        )
                 
        return trend_label, action, color, last_close, last_rsi, drawdown, reason, technical_target, potential_upside, technical_risk, potential_downside, w30, p30, w60, p60, w90, p90, confidence_score

    except Exception as e:
        logger.error(f"Errore generale in evaluate_strategy_full: {e}") 
        return "ERR", "Errore", "#eee", 0, 0, 0, str(e), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

# --- STRATEGIA PORTAFOGLIO AVANZATA ---
def generate_portfolio_advice(df, avg_price, current_price):
    """
    Advisor operativo avanzato.
    Ritorna: title, advice, color, trailing_stop, risk_score
    """
    # 1. CONTROLLO INTEGRITÀ DATI
    required = {'RSI', 'SMA_200', 'ATR', 'High', 'Volume', 'Close'}
    if df.empty or not required.issubset(df.columns):
        return "✋ DATI INSUFFICIENTI", "Impossibile calcolare strategia.", "#eeeeee", 0.0, 0

    if avg_price <= 0:
        avg_price = current_price # Evita division by zero, assume ingresso ora

    last = df.iloc[-1]
    rsi = last['RSI']
    sma = last['SMA_200']
    atr = last['ATR']
    vol_now = last['Volume']
    
    # ======================
    # CONTESTO & DATI
    # ======================
    pnl_pct = ((current_price - avg_price) / avg_price) * 100
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

    # DEFINIZIONE TREND DINAMICA (Miglioramento)
    # Usiamo l'ATR per capire se la distanza dalla media è significativa
    distanza_sma_pct = (current_price - sma) / sma * 100
    soglia_trend = max(1.0, atr_pct * 0.5) # Minimo 1% o metà della volatilità
    
    if distanza_sma_pct > soglia_trend:
        trend = "BULL"
    elif distanza_sma_pct < -soglia_trend:
        trend = "BEAR"
    else:
        trend = "SIDE" # Laterale

    # VOLUMI (Conferma istituzionale)
    # Usiamo la mediana invece della media per evitare che un singolo giorno anomalo sballi tutto
    vol_avg = df['Volume'].rolling(20).median().iloc[-1]
    has_volume = vol_now > (vol_avg * 1.25) # +25% sopra la mediana

    # TRAILING STOP (Chandelier Exit)
    rolling_high = df['High'].rolling(22).max().iloc[-1]
    # Se l'asset è molto volatile, diamo più aria allo stop (3.5 ATR), altrimenti 3
    atr_multiplier = 3.5 if atr_pct > 3.0 else 3.0
    raw_stop = rolling_high - (atr_multiplier * atr)
    # Lo stop non può mai essere sopra il prezzo attuale (sicurezza matematica)
    min_distance = current_price * 0.01  # almeno 1%
    trailing_stop = min(raw_stop, current_price - max(2.0 * atr, min_distance))

    # RISK SCORE (0-10)
    # Base: Volatilità. Malus: Trend Bear. Bonus: Trend Bull.
    base_risk = min(10, atr_pct * 1.5)
    if trend == "BEAR": base_risk += 2
    if rsi > 75 or rsi < 25: base_risk += 1
    risk_score = round(min(10, max(1, base_risk)), 1)

    # CONDIZIONI SPECIALI
    time_in_range = (
        df['Close'].tail(20).max() - df['Close'].tail(20).min()
    ) / current_price * 100
    
    very_stable = atr_pct < 1.5 and time_in_range < 4
    
    # SOGLIE DINAMICHE P&L
    t_low = max(3.0, 1.5 * atr_pct)
    t_mid = max(10.0, 4 * atr_pct)
    t_high = max(25.0, 8 * atr_pct)
    if len(df) >= 5:
        recent_return = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
    else:
        recent_return = 0

    # ==================================================
    # 🧠 DECISION ENGINE
    # ==================================================

    # 🚨 1. PERICOLO ESTREMO (Priorità Massima)
    if trend == "BEAR" and pnl_pct < -t_low and has_volume and recent_return < -5:
        title = "🔪 PERICOLO – COLTELLO CHE CADE"
        advice = (f"Trend ribassista confermato con volumi in aumento (vendite istituzionali). "
                  f"Non mediare ora. Rischio di ulteriori ribassi alto. "
                  f"Stop Loss tecnico suggerito: ${trailing_stop:.2f}.")
        color = "#ffebee" # Rosso molto chiaro

    # 💰 2. INCASSO PROFITTI (Trend Rotto)
    elif pnl_pct > t_high and trend == "BEAR":
        title = "🚨 INCASSA TUTTO (Trend Rotto)"
        advice = (f"Performance eccezionale (+{pnl_pct:.1f}%) ma il trend di lungo periodo è stato violato. "
                  f"Non restituire i profitti al mercato. Chiudi la posizione.")
        color = "#ffcccb"

    # 🛡️ 3. DIFESA (Profitto a rischio)
    elif pnl_pct > t_mid and trend != "BULL":
        title = "🛡️ PROTEGGI IL BOTTINO"
        advice = (f"Ottimo guadagno (+{pnl_pct:.1f}%) in un contesto che si sta indebolendo. "
                  f"Valuta di stringere lo stop a ${trailing_stop:.2f} o vendere parzialmente.")
        color = "#fff9c4" # Giallo

    # 🚀 4. ESECUZIONE TREND (Il caso migliore)
    elif pnl_pct > t_mid and trend == "BULL":
        if rsi > 75:
            title = "💰 TAKE PROFIT PARZIALE (Euforia)"
            advice = (f"Il prezzo corre (+{pnl_pct:.1f}%) ma l'RSI è molto alto ({rsi:.0f}). "
                      f"Vendi un 20-30% per sicurezza e lascia correre il resto con stop a ${trailing_stop:.2f}.")
            color = "#ffe0b2" # Arancione chiaro
        else:
            title = "🚀 MOONBAG – LASCIA CORRERE"
            advice = (f"Sei sul cavallo vincente (+{pnl_pct:.1f}%). Trend sano e volumi stabili. "
                      f"Non vendere nulla. Aggiorna solo lo Stop Loss a ${trailing_stop:.2f}.")
            color = "#c8f7c5" # Verde brillante

    # 💎 5. ACCUMULO INTELLIGENTE
    elif trend == "BULL" and pnl_pct < -2.0:
        if rsi < 40 and has_volume:
            title = "💎 ACCUMULO (Strong Buy)"
            advice = f"Il prezzo ritraccia con volumi alti: mani forti stanno comprando lo sconto. Ottima occasione per mediare (DCA)."
            color = "#b9f6ca"
        elif rsi < 45:
            title = "🛒 ACCUMULO CAUTO"
            advice = f"Ritracciamento fisiologico in trend rialzista. Si può accumulare piccole quote."
            color = "#e8f5e9"
        else:
            title = "✋ HOLD (Attendi)"
            advice = f"Leggera flessione. Non fare nulla per ora, attendi livelli di supporto migliori."
            color = "#f5f5f5"

    # 🧊 6. COSTO OPPORTUNITÀ
    elif very_stable and trend == "SIDE":
        title = "🧊 ASSET STABILE (Costo Opportunità)"
        advice = (f"L'asset si muove pochissimo (volatilità {atr_pct:.1f}%). "
                  f"Il capitale è bloccato. Considera di spostare questi fondi su asset più performanti.")
        color = "#e1f5fe" # Azzurro ghiaccio

    # ⚠️ 7. DEBOLEZZA GENERICA
    elif trend == "BEAR":
        title = "⚠️ MONITORARE (Trend Debole)"
        advice = f"Siamo sotto la media a 200 periodi. La struttura è fragile. Evita di aumentare l'esposizione."
        color = "#fff3e0"

    # 😴 8. DEFAULT
    else:
        title = "😴 MANTIENI"
        advice = "Nessun segnale operativo rilevante. Il prezzo sta consolidando. Mantieni la posizione."
        color = "#f5f5f5"

    return title, advice, color, trailing_stop, risk_score

# --- STORICO PORTAFOGLIO MIGLIORATO ---
def get_historical_portfolio_value(transactions, market_data_history):
    """
    Ricostruzione storico con gestione vendite completa.
    """
    if not transactions:
        return pd.DataFrame()

    df_tx = pd.DataFrame(transactions, columns=['id', 'symbol', 'qty', 'price', 'date', 'type', 'fee'])
    df_tx['date'] = pd.to_datetime(df_tx['date'])
    df_tx = df_tx.sort_values('date')

    start_date = df_tx['date'].min()
    end_date = pd.Timestamp.today().normalize()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    unique_tickers = df_tx['symbol'].unique()
    
    portfolio_state = {t: {'qty': 0.0, 'total_cost_basis': 0.0} for t in unique_tickers}
    history_records = []

    tx_grouped = df_tx.groupby('date')

    # Scarica prezzi
    price_history = pd.DataFrame(index=date_range)
    for t in unique_tickers:
        if t in market_data_history:
            price_history[t] = market_data_history[t]['Close'].reindex(date_range).ffill().bfill()
        else:
            price_history[t] = 0.0

    # Loop giornaliero
    for current_date in date_range:
        # Applica transazioni
        if current_date in tx_grouped.groups:
            daily_txs = tx_grouped.get_group(current_date)
            for _, tx in daily_txs.iterrows():
                sym = tx['symbol']
                q = tx['qty']
                p = tx['price']
                f = tx['fee']
                
                curr_qty = portfolio_state[sym]['qty']
                curr_cost = portfolio_state[sym]['total_cost_basis']

                if tx['type'] == 'BUY':
                    portfolio_state[sym]['qty'] += q
                    portfolio_state[sym]['total_cost_basis'] += (q * p) + f
                
                elif tx['type'] == 'SELL':
                    if curr_qty > 0:
                        avg_cost = curr_cost / curr_qty
                        cost_removed = avg_cost * q
                        portfolio_state[sym]['qty'] = max(0, curr_qty - q)
                        portfolio_state[sym]['total_cost_basis'] = max(0, curr_cost - cost_removed)

        # Calcola valori
        daily_total_value = 0.0
        daily_total_invested = 0.0
        asset_values = {}

        for sym in unique_tickers:
            qty = portfolio_state[sym]['qty']
            cost_basis = portfolio_state[sym]['total_cost_basis']
            market_price = price_history.at[current_date, sym]
            
            curr_val = qty * market_price
            
            daily_total_value += curr_val
            daily_total_invested += cost_basis
            asset_values[sym] = curr_val

        record = {
            'Total Value': daily_total_value,
            'Total Invested': daily_total_invested,
            **asset_values
        }
        history_records.append(record)

    df_history = pd.DataFrame(history_records, index=date_range)
    return df_history

def generate_enhanced_excel_report(df_hist, current_portfolio, transactions_list=None):
    """
    Genera un Excel professionale con:
    1. Foglio 'Dati Storici': Tabella completa con valori, investito, utile e % giornaliera.
    2. Foglio 'Portafoglio': Situazione attuale asset per asset.
    3. Foglio 'Dashboard Grafici': Solo grafici riassuntivi.
    4. Foglio 'Transazioni': Registro completo operazioni.
    """
    from io import BytesIO
    import pandas as pd
    
    output = BytesIO()
    
    # Pre-calcoli sui dati storici
    if 'Total Invested' not in df_hist.columns:
        df_hist['Total Invested'] = 0.0
    
    df_hist['Utile Netto (€)'] = df_hist['Total Value'] - df_hist['Total Invested']
    df_hist['Performance %'] = df_hist['Total Value'].pct_change().fillna(0)
    
    cols_main = ['Total Value', 'Total Invested', 'Utile Netto (€)', 'Performance %']
    cols_assets = [c for c in df_hist.columns if c not in cols_main]
    df_final_hist = df_hist[cols_main + cols_assets]

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- DEFINIZIONE FORMATI ---
        fmt_currency = workbook.add_format({'num_format': '€ #,##0.00'})
        fmt_pct = workbook.add_format({'num_format': '0.00%'})
        
        fmt_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'num_format': '0.00%'})
        fmt_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'num_format': '0.00%'})
        
        fmt_curr_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'num_format': '€ #,##0.00'})
        fmt_curr_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'num_format': '€ #,##0.00'})
        
        fmt_txt_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'bold': True})
        fmt_txt_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'bold': True})

        # ==========================================
        # FOGLIO 1: DATI STORICI
        # ==========================================
        sheet_hist_name = 'Dati Storici'
        df_final_hist.to_excel(writer, sheet_name=sheet_hist_name, index=True)
        ws_hist = writer.sheets[sheet_hist_name]
        
        last_row = len(df_final_hist) + 1
        
        ws_hist.set_column(0, 0, 15) 
        ws_hist.set_column(1, 3, 20, fmt_currency) 
        ws_hist.set_column(4, 4, 15, fmt_pct)
        ws_hist.set_column(5, len(df_final_hist.columns), 15, fmt_currency)
        
        if len(df_final_hist) > 0:
            ws_hist.conditional_format(1, 4, last_row, 4, {'type': 'cell', 'criteria': '>', 'value': 0, 'format': fmt_green})
            ws_hist.conditional_format(1, 4, last_row, 4, {'type': 'cell', 'criteria': '<', 'value': 0, 'format': fmt_red})
            ws_hist.conditional_format(1, 3, last_row, 3, {'type': 'cell', 'criteria': '>', 'value': 0, 'format': fmt_curr_green})
            ws_hist.conditional_format(1, 3, last_row, 3, {'type': 'cell', 'criteria': '<', 'value': 0, 'format': fmt_curr_red})

        # ==========================================
        # FOGLIO 2: PORTAFOGLIO ATTUALE
        # ==========================================
        sheet_pf_name = 'Portafoglio'
        data_pf = []
        
        for k, v in current_portfolio.items():
            price_now = v.get('cur_price', v.get('avg_price', 0.0))
            qty = v.get('qty', 0.0)
            pnl_val = v.get('pnl_pct', 0.0)

            data_pf.append({
                "Asset": k,
                "Quantità": qty,
                "Prezzo Medio": v.get('avg_price', 0.0),
                "Prezzo Attuale": price_now,
                "Valore Totale": qty * price_now,
                "P&L %": pnl_val / 100
            })
        
        if data_pf:
            df_pf = pd.DataFrame(data_pf)
            df_pf.to_excel(writer, sheet_name=sheet_pf_name, index=False)
            ws_pf = writer.sheets[sheet_pf_name]
            
            ws_pf.set_column('A:A', 15) 
            ws_pf.set_column('B:B', 12) 
            ws_pf.set_column('C:E', 18, fmt_currency) 
            ws_pf.set_column('F:F', 12, fmt_pct) 
            
            ws_pf.conditional_format(1, 5, len(df_pf), 5, {'type': 'cell', 'criteria': '>', 'value': 0, 'format': fmt_green})
            ws_pf.conditional_format(1, 5, len(df_pf), 5, {'type': 'cell', 'criteria': '<', 'value': 0, 'format': fmt_red})

        # ==========================================
        # FOGLIO 3: DASHBOARD GRAFICI
        # ==========================================
        sheet_dash_name = 'Dashboard Grafici'
        ws_dash = workbook.add_worksheet(sheet_dash_name)
        ws_dash.hide_gridlines(2)
        ws_dash.write('B2', "Report Finanziario - InvestAI", workbook.add_format({'bold': True, 'font_size': 18, 'font_color': '#004d40'}))

        if len(df_final_hist) > 0:
            # Grafico Evoluzione
            chart_evo = workbook.add_chart({'type': 'line'})
            chart_evo.add_series({
                'name':       'Valore Portafoglio',
                'categories': f"='{sheet_hist_name}'!$A$2:$A${last_row}",
                'values':     f"='{sheet_hist_name}'!$B$2:$B${last_row}",
                'line':       {'color': '#004d40', 'width': 2.5}
            })
            chart_evo.add_series({
                'name':       'Capitale Investito',
                'categories': f"='{sheet_hist_name}'!$A$2:$A${last_row}",
                'values':     f"='{sheet_hist_name}'!$C$2:$C${last_row}",
                'line':       {'color': '#ef5350', 'width': 1.5, 'dash_type': 'dash'}
            })
            chart_evo.set_title({'name': 'Crescita del Capitale'})
            chart_evo.set_size({'width': 800, 'height': 400})
            ws_dash.insert_chart('B4', chart_evo)

            # Grafico Utile Netto
            chart_pnl = workbook.add_chart({'type': 'column'})
            chart_pnl.add_series({
                'name':       'Utile Netto',
                'categories': f"='{sheet_hist_name}'!$A$2:$A${last_row}",
                'values':     f"='{sheet_hist_name}'!$D$2:$D${last_row}",
                'fill':       {'color': '#66bb6a'},
                'gap':        50
            })
            chart_pnl.set_title({'name': 'Andamento Utile Netto (€)'})
            chart_pnl.set_size({'width': 800, 'height': 350})
            ws_dash.insert_chart('B26', chart_pnl)

        if data_pf:
            chart_pie = workbook.add_chart({'type': 'doughnut'})
            chart_pie.add_series({
                'name': 'Allocazione',
                'categories': f"='{sheet_pf_name}'!$A$2:$A${len(df_pf)+1}",
                'values':     f"='{sheet_pf_name}'!$E$2:$E${len(df_pf)+1}",
                'data_labels': {'percentage': True, 'position': 'outside'}
            })
            chart_pie.set_title({'name': 'Allocazione Asset'})
            chart_pie.set_style(10)
            chart_pie.set_size({'width': 400, 'height': 350})
            ws_dash.insert_chart('O4', chart_pie)

        # ==========================================
        # FOGLIO 4: TRANSAZIONI
        # ==========================================
        if transactions_list:
            # Creazione DF da lista di tuple/dizionari
            # La struttura attesa è tuple: (id, symbol, qty, price, date, type, fee)
            df_tx = pd.DataFrame(transactions_list, columns=['ID', 'Asset', 'Qta', 'Prezzo', 'Data', 'Tipo', 'Fee'])
            
            # Calcolo Totale Transazione
            df_tx['Totale (€)'] = (df_tx['Qta'] * df_tx['Prezzo']) + df_tx['Fee']
            
            # Ordinamento e pulizia data
            df_tx['Data'] = pd.to_datetime(df_tx['Data']).dt.strftime('%Y-%m-%d')
            df_tx.sort_values('Data', ascending=False, inplace=True)
            
            # Scrittura
            sheet_tx_name = 'Transazioni'
            df_tx.to_excel(writer, sheet_name=sheet_tx_name, index=False)
            ws_tx = writer.sheets[sheet_tx_name]
            
            # Larghezze
            ws_tx.set_column('A:A', 5)  # ID
            ws_tx.set_column('B:B', 10) # Asset
            ws_tx.set_column('C:C', 10) # Qta
            ws_tx.set_column('D:D', 12, fmt_currency) # Prezzo
            ws_tx.set_column('E:E', 12) # Data
            ws_tx.set_column('F:F', 8)  # Tipo
            ws_tx.set_column('G:G', 10, fmt_currency) # Fee
            ws_tx.set_column('H:H', 15, fmt_currency) # Totale
            
            # Formattazione Condizionale (Verde per BUY, Rosso per SELL)
            # Colonna F (Tipo) è indice 5 (0-based)
            tx_len = len(df_tx) + 1
            ws_tx.conditional_format(1, 5, tx_len, 5, {
                'type': 'cell',
                'criteria': '==',
                'value': '"BUY"',
                'format': fmt_txt_green
            })
            ws_tx.conditional_format(1, 5, tx_len, 5, {
                'type': 'cell',
                'criteria': '==',
                'value': '"SELL"',
                'format': fmt_txt_red
            })

    return output.getvalue()

def estimate_days_to_target(current_price, target_price, atr):
    """
    Stima i giorni necessari per raggiungere il target basandosi sulla volatilità (ATR).
    Formula: Distanza / (ATR * Fattore Prudenza)
    """
    if atr <= 0 or current_price <= 0:
        return 0, "N/A"
    
    distance = abs(target_price - current_price)
    
    # Assumiamo che il prezzo non vada dritto al target ogni giorno.
    # Usiamo un "Fattore di Efficienza" del 40% (cioè il prezzo zig-zaga).
    daily_move_effective = atr * 0.4
    
    days = int(distance / daily_move_effective)
    
    # Limiti logici per evitare numeri assurdi
    if days < 1: days = 1
    if days > 120: return days, "> 4 Mesi"
    if days > 60: return days, f"~ {int(days/30)} Mesi"
    if days > 30: return days, f"~ 1 Mese e mezzo"
    
    return days, f"~ {days} Giorni"









