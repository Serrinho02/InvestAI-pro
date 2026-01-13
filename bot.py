import telebot
import schedule
import time
import threading
from datetime import datetime

from logic import (
    DBManager, 
    get_data_raw, 
    evaluate_strategy_full, 
    generate_portfolio_advice, 
    AUTO_SCAN_TICKERS, 
    TELEGRAM_BOT_TOKEN
)

# Inizializza bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
db = DBManager()

print("🤖 InvestAI Bot inizializzato...")

# --- FUNZIONI UTILITÀ ---

def get_verified_user(message):
    """Verifica utente registrato."""
    chat_id = str(message.chat.id)
    website_username = db.get_user_by_chat_id(chat_id)
    return website_username, chat_id

def get_market_data_for_user(username):
    """Scarica dati per utente specifico."""
    pf, _ = db.get_portfolio_summary(username)
    tickers = set(AUTO_SCAN_TICKERS)
    if pf:
        tickers.update(pf.keys())
    return get_data_raw(list(tickers)), pf

# --- COMANDI ---

@bot.message_handler(commands=['start'])
def send_welcome(message):
    website_user, chat_id = get_verified_user(message)
    
    if website_user:
        db.save_chat_id(website_user, chat_id)
        
        msg = (f"✅ **Benvenuto {website_user}!**\n\n"
               f"Notifiche automatiche **ATTIVE** (08:00 UTC).\n\n"
               f"**Comandi:**\n"
               f"/portafoglio - Analisi istantanea\n"
               f"/mercato - Scansione opportunità\n"
               f"/stop - Disattiva report automatico")
        bot.reply_to(message, msg, parse_mode="Markdown")
    else:
        bot.reply_to(message, 
                    f"⛔ **Utente non riconosciuto**\n\n"
                    f"Chat ID: `{chat_id}`\n\n"
                    f"Inseriscilo nelle impostazioni del sito.", 
                    parse_mode="Markdown")

@bot.message_handler(commands=['stop'])
def stop_notifications(message):
    website_user, chat_id = get_verified_user(message)
    
    if website_user:
        if db.disable_notifications(website_user, chat_id):
            bot.reply_to(message, 
                        "🔕 **Report automatico DISATTIVATO.**\n\n"
                        "Puoi comunque usare:\n"
                        "/portafoglio e /mercato manualmente.\n\n"
                        "Digita /start per riattivare.", 
                        parse_mode="Markdown")
        else:
            bot.reply_to(message, "Errore durante la disattivazione.")
    else:
        bot.reply_to(message, "Non sei registrato.")

@bot.message_handler(commands=['portafoglio'])
def send_portfolio(message):
    username, chat_id = get_verified_user(message)
    
    if not username:
        bot.send_message(chat_id, 
                         "⚠️ Non sei configurato. Collega l'account sul sito.", 
                         parse_mode="Markdown")
        return

    bot.send_message(chat_id, 
                     f"⏳ Analisi portafoglio di **{username}**...", 
                     parse_mode="Markdown")
    
    market_data, pf = get_market_data_for_user(username)
    
    if not pf:
        bot.send_message(chat_id, "Portafoglio vuoto.")
        return

    msgs = []
    for t, data in pf.items():
        if t in market_data:
            cur_price = market_data[t]['Close'].iloc[-1]
            
            # FIX: Gestione 5 valori di ritorno (title, advice, color, trailing_stop, risk_score)
            res = generate_portfolio_advice(market_data[t], data['avg_price'], cur_price)
            if len(res) == 5:
                tit, adv, _, t_stop, risk = res
            else:
                tit, adv, _, t_stop = res # Fallback
                risk = 0

            # Dati Tecnici
            tl, act, col, pr, rsi, dd, reason, tgt, pot, risk_pr, risk_pot, w30, p30, w60, p60, w90, p90, conf = evaluate_strategy_full(market_data[t])
            
            pnl = ((cur_price - data['avg_price']) / data['avg_price']) * 100
            emoji = "🟢" if pnl > 0 else "🔴"
            pot_str = f"+{pot:.1f}%" if pot > 0 else f"{pot:.1f}%"
            
            msg = (f"{emoji} <b>{t}</b>: {pnl:+.1f}%\n"
                   f"📢 <b>{tit}</b>\n"
                   f"🛡️ Stop Dinamico: <b>${t_stop:.2f}</b>\n" # NUOVO
                   f"🎯 Target: ${tgt:.2f} (<b>{pot_str}</b>)\n"
                   f"🏆 Score: <b>{conf}/100</b>\n"
                   f"<i>{adv}</i>")
            msgs.append(msg)
            
    if msgs:
        full_msg = f"💼 <b>PORTAFOGLIO DI {username.upper()}</b>\n\n" + "\n\n".join(msgs)
        try:
            bot.send_message(chat_id, full_msg, parse_mode="HTML")
        except Exception as e:
            bot.send_message(chat_id, "Errore invio dati.")
            print(f"Errore invio portafoglio: {e}")
    else:
        bot.send_message(chat_id, "Impossibile scaricare dati.")

@bot.message_handler(commands=['mercato'])
def send_market_scan(message):
    username, chat_id = get_verified_user(message)
    
    bot.send_message(chat_id, "🔎 Scansione mercato...")
    
    market_data = get_data_raw(AUTO_SCAN_TICKERS)
    opportunities = []
    
    for t in AUTO_SCAN_TICKERS:
        if t in market_data:
            tl, act, col, pr, rsi, dd, reason, tgt, pot, risk_pr, risk_pot, w30, p30, w60, p60, w90, p90, conf = evaluate_strategy_full(market_data[t])
            
            if "ORO" in act or ("ACQUISTA" in act and conf >= 40):
                icon = "💎" if "ORO" in act else "🟢"
                pot_str = f"+{pot:.1f}%"
                msg = (f"{icon} <b>{t}</b>: {act}\n"
                       f"🏆 Score: <b>{conf}/100</b>\n"
                       f"Prezzo: ${pr:.2f} | RSI: {rsi:.0f}\n"
                       f"🎯 Upside: <b>{pot_str}</b>\n"
                       f"<i>{reason}</i>")
                opportunities.append((conf, msg))
    
    if opportunities:
        # Ordina per confidence score
        opportunities.sort(reverse=True, key=lambda x: x[0])
        msgs = [m for _, m in opportunities[:10]]  # Top 10
        
        full_msg = "🚀 <b>TOP OPPORTUNITÀ</b>\n\n" + "\n\n".join(msgs)
        bot.send_message(chat_id, full_msg, parse_mode="HTML")
    else:
        bot.send_message(chat_id, "Nessuna occasione rilevata.")

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """
📱 <b>COMANDI DISPONIBILI</b>

/start - Attiva notifiche automatiche
/portafoglio - Analisi del tuo portafoglio
/mercato - Scansiona nuove opportunità
/stop - Disattiva report giornaliero
/help - Mostra questo messaggio

<i>💡 Suggerimento: Configura il tuo Chat ID sul sito per ricevere consigli automatici ogni mattina.</i>
    """
    bot.send_message(message.chat.id, help_text, parse_mode="HTML")

# --- JOB GIORNALIERO ---

def send_daily_report():
    """Report automatico mattutino."""
    print(f"--- ⏰ Report Giornaliero: {datetime.now()} ---")
    
    try:
        users = db.get_users_with_telegram()
        
        if not users:
            print("-> Nessun utente attivo.")
            return
            
        print(f"-> {len(users)} utenti attivi.")

        # Scarica dati unici
        all_tickers = set(AUTO_SCAN_TICKERS)
        user_map = {} 
        
        for u, chat_id in users:
            pf, _ = db.get_portfolio_summary(u)
            user_map[u] = pf
            if pf: 
                all_tickers.update(pf.keys())
        
        print(f"-> Scaricamento {len(all_tickers)} asset...")
        market_data = get_data_raw(list(all_tickers))
        
        for username, chat_id in users:
            messages = []
            pf = user_map.get(username, {})
            
            # 1. Portafoglio urgente
            if pf:
                for t, data in pf.items():
                    if t in market_data:
                        cur_price = market_data[t]['Close'].iloc[-1]
                        
                        # FIX: Unpacking sicuro
                        res = generate_portfolio_advice(market_data[t], data['avg_price'], cur_price)
                        tit = res[0] # Prendiamo solo il titolo
                        
                        keywords_urgenti = ["VENDI", "INCASSA", "PROTEGGI", "MOONBAG", "COLTELLO"]
                        if any(k in tit for k in keywords_urgenti):
                             pnl = ((cur_price - data['avg_price']) / data['avg_price']) * 100
                             messages.append(f"🚨 <b>{t}</b> ({pnl:+.1f}%): {tit}")

            # 2. Mercato (Golden + High Confidence)
            for t in AUTO_SCAN_TICKERS:
                if t in market_data and (not pf or t not in pf):
                    _, act, _, _, _, _, _, _, pot, _, _, _, _, _, _, _, _, conf = evaluate_strategy_full(market_data[t])
                    if "ORO" in act:
                        messages.append(f"💎 <b>{t} - GOLDEN!</b> (+{pot:.1f}%) [Score: {conf}]")
                    elif "ACQUISTA" in act and conf >= 60:
                        messages.append(f"🟢 <b>{t} - BUY</b> (+{pot:.1f}%) [Score: {conf}]")
            
            if messages:
                full_msg = f"🌅 <b>Buongiorno {username}!</b>\n\n" + "\n\n".join(messages[:8])
                try:
                    bot.send_message(chat_id, full_msg, parse_mode="HTML")
                    print(f"-> Inviato a {username}")
                except Exception as e:
                    print(f"-> Errore invio a {username}: {e}")
            else:
                print(f"-> Nessuna novità per {username}")
                
    except Exception as e:
        print(f"ERROR in Daily Report: {e}")

# --- SCHEDULER ---

def run_scheduler():
    """Scheduler per report giornaliero."""
    schedule.every().day.at("08:00").do(send_daily_report)
    print(f"🕒 Scheduler attivo. Orario: {datetime.now().strftime('%H:%M')}")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            print(f"Scheduler Error: {e}")
            time.sleep(60)
