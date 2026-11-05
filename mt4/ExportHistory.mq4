//+------------------------------------------------------------------+
//|                                              ExportHistory.mq4   |
//|  Exports OHLCV history for the symbols/timeframes you analyse    |
//|  into <MT4 data folder>\MQL4\Files\<SYMBOL>_<minutes>.csv        |
//|                                                                  |
//|  Install: MetaEditor -> File -> Open Data Folder -> MQL4\Scripts |
//|           copy this file there, compile (F7), then in MT4:       |
//|           Navigator -> Scripts -> ExportHistory -> double click  |
//+------------------------------------------------------------------+
#property strict
#property script_show_inputs

input string Symbols    = "XAUUSD,EURUSD,BTCUSD";  // comma separated
input string Timeframes = "1,5,15,60,240";         // minutes, comma separated

ENUM_TIMEFRAMES MinutesToPeriod(int minutes)
{
   switch(minutes)
   {
      case 1:    return PERIOD_M1;
      case 5:    return PERIOD_M5;
      case 15:   return PERIOD_M15;
      case 30:   return PERIOD_M30;
      case 60:   return PERIOD_H1;
      case 240:  return PERIOD_H4;
      case 1440: return PERIOD_D1;
      case 10080:return PERIOD_W1;
      default:   return PERIOD_CURRENT;
   }
}

string Trim(string text)
{
   StringTrimLeft(text);
   StringTrimRight(text);
   return text;
}

void OnStart()
{
   string symbols[];
   string timeframes[];
   int symbolCount = StringSplit(Symbols, ',', symbols);
   int timeframeCount = StringSplit(Timeframes, ',', timeframes);
   int written = 0;

   for(int s = 0; s < symbolCount; s++)
   {
      string symbol = Trim(symbols[s]);
      if(StringLen(symbol) == 0)
         continue;
      if(!SymbolSelect(symbol, true))
      {
         PrintFormat("Skipping %s - not in Market Watch", symbol);
         continue;
      }

      for(int t = 0; t < timeframeCount; t++)
      {
         int minutes = (int)StringToInteger(Trim(timeframes[t]));
         ENUM_TIMEFRAMES period = MinutesToPeriod(minutes);
         if(period == PERIOD_CURRENT)
         {
            PrintFormat("Skipping %s - unsupported timeframe %d", symbol, minutes);
            continue;
         }

         int total = iBars(symbol, period);
         if(total <= 0)
         {
            PrintFormat("No history yet for %s %dm - open that chart once and retry",
                        symbol, minutes);
            continue;
         }

         string filename = symbol + "_" + IntegerToString(minutes) + ".csv";
         int handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
         if(handle == INVALID_HANDLE)
         {
            PrintFormat("Could not open %s (error %d)", filename, GetLastError());
            continue;
         }

         FileWrite(handle, "time", "open", "high", "low", "close", "volume");
         int digits = (int)MarketInfo(symbol, MODE_DIGITS);
         for(int i = total - 1; i >= 0; i--)
         {
            FileWrite(handle,
                      TimeToString(iTime(symbol, period, i), TIME_DATE | TIME_MINUTES),
                      DoubleToString(iOpen(symbol, period, i), digits),
                      DoubleToString(iHigh(symbol, period, i), digits),
                      DoubleToString(iLow(symbol, period, i), digits),
                      DoubleToString(iClose(symbol, period, i), digits),
                      DoubleToString((double)iVolume(symbol, period, i), 0));
         }
         FileClose(handle);
         written++;
         PrintFormat("Wrote %s (%d bars)", filename, total);
      }
   }

   PrintFormat("Done - %d file(s) written to MQL4\\Files", written);
   Alert("ExportHistory finished: ", written, " file(s) written.");
}
//+------------------------------------------------------------------+
