import MetaTrader5 as mt5
import time

def print_trade_info(pos):
    print(f"\n=== Trying to close {pos.type} {pos.volume} {pos.symbol} ticket {pos.ticket} at entry {pos.price_open} ===")
    print(f"  Symbol: {pos.symbol}, Volume: {pos.volume}, Open price: {pos.price_open}, SL: {pos.sl}, TP: {pos.tp}")
    info = mt5.symbol_info(pos.symbol)
    print(f"  Symbol Info: {info}")

def try_close_position(pos):
    symbol = pos.symbol
    ticket = pos.ticket
    volume = pos.volume
    digits = mt5.symbol_info(symbol).digits
    allowed_filling = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    allowed_deviation = [100, 300, 500, 1000, 2000]
    price_funcs = [lambda: 0,  # MT5 doc: 0 means market
                   lambda: mt5.symbol_info_tick(symbol).bid if pos.type == 1 else mt5.symbol_info_tick(symbol).ask]
    # 0 = buy, 1 = sell
    close_type = mt5.ORDER_TYPE_BUY if pos.type == 1 else mt5.ORDER_TYPE_SELL

    for price_func in price_funcs:
        try:
            price = round(price_func(), digits) if callable(price_func) else price_func
        except Exception:
            price = 0
        for filling in allowed_filling:
            for deviation in allowed_deviation:
                print(f"Trying to close {symbol} ticket={ticket} volume={volume} at price={price}, filling={filling}, deviation={deviation}")
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": close_type,
                    "position": ticket,
                    "price": price,
                    "deviation": deviation,
                    "magic": 42,
                    "comment": "close_test",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                }
                res = mt5.order_send(req)
                retcode = getattr(res, "retcode", None)
                comment = getattr(res, "comment", None)
                print(f"Result: retcode={retcode}, comment={comment}")
                if retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"SUCCESS: Trade closed with price={price}, filling={filling}, deviation={deviation}")
                    return True
                time.sleep(0.5)
    print(f"FAILED: Could not close position {ticket}")
    return False

def close_all_open_positions(symbol_filter=None):
    if not mt5.initialize():
        print("MT5 init failed.")
        return

    positions = mt5.positions_get()
    if not positions:
        print("No open positions.")
        return

    for pos in positions:
        if symbol_filter and pos.symbol != symbol_filter:
            continue
        print_trade_info(pos)
        try_close_position(pos)

    mt5.shutdown()

if __name__ == "__main__":
    # You can pass a symbol like 'XAUUSD' to only close those positions, or None for all
    close_all_open_positions(symbol_filter=None)
