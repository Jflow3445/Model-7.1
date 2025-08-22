import MetaTrader5 as mt5
import time

SYMBOL = "EURUSD"
VOLUME = 0.01
PRICE_OFFSET = 0.0005

filling_modes = {
    mt5.ORDER_FILLING_IOC: "IOC",
    mt5.ORDER_FILLING_RETURN: "RETURN",
    mt5.ORDER_FILLING_FOK: "FOK",
}

def try_filling_mode(symbol, filling_mode):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"Failed to fetch tick for {symbol}")
        return False

    price = tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": VOLUME,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price + PRICE_OFFSET,  # slightly above ask
        "sl": price - 10 * PRICE_OFFSET,
        "tp": price + 10 * PRICE_OFFSET,
        "deviation": 10,
        "magic": 999,
        "comment": f"filling_mode_test_{filling_modes[filling_mode]}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[✓] {filling_modes[filling_mode]} succeeded")
        return True
    else:
        print(f"[✗] {filling_modes[filling_mode]} failed: {result.retcode} — {result.comment}")
        return False

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    if not mt5.symbol_select(SYMBOL, True):
        print(f"Failed to select symbol {SYMBOL}")
        return

    print(f"Checking filling modes for {SYMBOL}...\n")
    supported = []
    for mode in filling_modes:
        if try_filling_mode(SYMBOL, mode):
            supported.append(filling_modes[mode])
        time.sleep(1)

    print("\nSupported filling modes:")
    for mode in supported:
        print(f" - {mode}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
