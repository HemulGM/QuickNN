unit QuickNN.Timing;

interface

const
  CLOCKS_PER_SEC = 1000000000;
  CLOCKS_PER_MS = 1000000;
  CLOCKS_PER_US = 1000;

// Monotonic high-resolution time in nanoseconds, for elapsed-time measurements.
function Clock: Int64;

implementation

uses
  System.Diagnostics;

function Clock: Int64;
begin
  Result := Trunc(TStopwatch.GetTimeStamp / (TStopwatch.Frequency / CLOCKS_PER_SEC));
end;

end.
