"""
Canonical ETF universe for EMA crossover backtest.
Broad coverage: equities, sectors, commodities, bonds, currencies.
"""

EQUITY_BROAD = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "RSP": "Invesco S&P 500 Equal Weight",
    "SCHD": "Schwab US Dividend Equity",
    "BRK/B": "Berkshire Hathaway B",
    "VT": "Vanguard Total World Stock",
}

SECTOR_SPDR = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

SECTOR_ISTOCKS = {
    "IYW": "Tech (iShares)",
    "IYH": "Healthcare (iShares)",
    "IYF": "Financials (iShares)",
    "IYC": "Consumer (iShares)",
    "IYE": "Energy (iShares)",
    "IYJ": "Industrials (iShares)",
}

INTERNATIONAL = {
    "VXUS": "Vanguard Total International Stock",
    "EEM": "Emerging Markets",
    "EWH": "Hong Kong",
    "FXI": "China",
}

COMMODITIES = {
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil",
    "DBC": "Commodities (broad)",
    "DBB": "Metals (broad)",
    "UNG": "Natural Gas",
    "WPM": "Wheaton Precious Metals",
}

BONDS = {
    "BND": "US Aggregate Bonds",
    "TLT": "20+ Year Treasury",
    "SHV": "1-3 Year Treasury",
    "HYG": "High Yield Corporate",
    "LQD": "Investment Grade Corporate",
}

CURRENCIES = {
    "FXE": "Euro",
    "FXY": "Japanese Yen",
    "FXB": "British Pound",
    "FXA": "Australian Dollar",
    "CurrencyShares": "Multiple",
}

# Alternative/growth
GROWTH = {
    "TQQQ": "Nasdaq 100 3x Leverage",
    "UPRO": "S&P 500 3x Leverage",
    "SQQQ": "Nasdaq 100 -3x (bearish)",
    "BTC": "Bitcoin",
    "CHAT": "Chatbot/AI ETF",
}

# Aggregate all into canonical universe
UNIVERSE = {
    **EQUITY_BROAD,
    **SECTOR_SPDR,
    **SECTOR_ISTOCKS,
    **INTERNATIONAL,
    **COMMODITIES,
    **BONDS,
    **CURRENCIES,
    **GROWTH,
}

# Primary universe for first backtest: equities + sectors only (lower correlation, easier to manage)
PRIMARY_UNIVERSE = {
    **EQUITY_BROAD,
    **SECTOR_SPDR,
}

if __name__ == "__main__":
    print(f"Full universe: {len(UNIVERSE)} ETFs")
    print(f"Primary universe: {len(PRIMARY_UNIVERSE)} ETFs")
    print("\nPrimary ETFs:")
    for ticker, description in sorted(PRIMARY_UNIVERSE.items()):
        print(f"  {ticker:6s} — {description}")
