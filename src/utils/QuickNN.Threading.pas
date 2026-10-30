{ <Steroids : (Thread Pooling wrapper for Delphi) >
  Copyright (c) <2022> <Haitham Shatti  <haitham.shatti at gmail dot com> >
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
  USE OR OTHER DEALINGS IN THE SOFTWARE.
}

unit QuickNN.Threading;

interface
uses
  System.SysUtils,
  System.Classes,
  System.Math,
  System.TypInfo
{$IFDEF MSWINDOWS}
  ,
  Winapi.Windows
{$ENDIF}
  ,
  System.SyncObjs;

const
  DEFAULT_STACK_SIZE = 4 * 1024 * 1024;
type
{$IF not defined(PtrInt)}
  PPtrInt = ^PtrInt;
  PtrInt = IntPtr;
{$ENDIF}
  PMPParams = ^TMPParams;
  TMPParams = record
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U: Pointer;
  end;
  TMPParamsArray = array[0..$80] of TMPParams;
{$IF defined(POSIX)}

{$ELSE}
  {$WARN SYMBOL_PLATFORM OFF} // This branch uses the Windows RTL priority type.
  TGroupPriority = TThreadPriority;
  {$WARN SYMBOL_PLATFORM DEFAULT}
{$ENDIF}
  TGroupProc = procedure(const RangeStart, RangeEnd: IntPtr; const Params: Pointer);
  TGroupProcNested =
      reference to procedure(const RangeStart, RangeEnd: IntPtr; const Params: Pointer); // must be [register]?
  TGroupMethod = procedure(const RangeStart, RangeEnd: IntPtr; const Params: Pointer) of object;
  TThreadProc = procedure(Arg: Pointer);
  TThreadMeth = procedure(Arg: Pointer) of object;
  TThreadNest = reference to procedure(Arg: Pointer);
  TThreadProc2 = procedure(I: IntPtr; Arg: Pointer);
  TThreadProcNested = reference to procedure(I: IntPtr; Arg: Pointer);
  TOPool = class;

  { TOThread }
  TOThread = class(TThread)
  // TGroupProc=procedure(const params:PPointer);
  private
    Fire: {$IFDEF MSWINDOWS} THandle {$ELSE} TEvent {$ENDIF};
    FStart, FSpan, FEnd: IntPtr;
    FID: Integer;
    FParams: Pointer;
    FSync: TThreadMethod;
    FProc: TGroupProc;
    FProcNested: TGroupProcNested;
    FMethod: TGroupMethod;
    FMethod2: TThreadMeth;
    FThreadProc: TThreadProc;
    FThreadNest: TThreadNest;
    FThreadProc2: TThreadProc2;
    FThreadProcNested: TThreadProcNested;
    FBusy: Boolean;
    FPool: TOPool;
  public
    procedure Execute; override;
    constructor Create(CreateSuspended: Boolean; const StackSize: NativeUInt = DEFAULT_STACK_SIZE); overload;
    constructor Create(const Proc: TGroupProc; const Params: Pointer); overload;
    destructor Destroy; override;
    property Pool: TOPool read FPool;
    property Busy: Boolean read FBusy;
    property Id: Integer read FID;
    property From: IntPtr read FStart;
    property &to: IntPtr read FEnd;
  end;

  { TOPool }
  TOPool = class
    type
      TCPUFreq = record
        Base, Max, Bus, Xtal: LongWord;
        Ratio: Double;
      end;
      TBuff = array[0..47] of AnsiChar;
    class var
      GlobalId: Integer;
  private
    Id: Integer;
    Pool: array of TOThread;
    OTCount: Integer;

    PoolDone: TEvent;

    procedure WaitForPool;
    function GetWorkers: LongInt;
  public
    constructor Create(PoolCount: Integer = 0);
    class function CPUName(): TBuff; static;
    class function CPUFreqMhz(): TCPUFreq; static;
    destructor Destroy; override;
    procedure SetWorkers(Count: LongInt);
    procedure &for(const Proc: TGroupProc; const LocalFrom, LocalTo: IntPtr; const Params: Pointer = nil); overload;

    procedure &for(
        const Proc: TGroupProcNested;
        const LocalFrom, LocalTo: IntPtr;
        const Params: Pointer = nil
    ); overload;

    procedure &for(const Proc: TGroupMethod; const LocalFrom, LocalTo: IntPtr; const Params: Pointer = nil); overload;

    procedure &for(
        const Proc: TThreadProc2;
        const Start, Count: IntPtr;
        const Args: Pointer;
        const Step: IntPtr = 1;
        const Sync: TThreadMethod = nil
    ); overload;

    procedure &for(
        const Proc: TThreadProcNested;
        const Start, Count: IntPtr;
        const Args: Pointer = nil;
        const Step: IntPtr = 1;
        const Sync: TThreadMethod = nil
    ); overload;

    function IsBusy: Boolean;
{$IFDEF POSIX}
    procedure SetPriority(const Priority: Integer);
{$ELSE}
    procedure SetPriority(const Priority: TGroupPriority);
{$ENDIF}
    property Workers: LongInt read GetWorkers write SetWorkers;
    property PendingWorkCount: Integer read OTCount;
  // FCriticalSection:TRTLCriticalSection;
  end;
var
  Mp, Mp2, Mp4, Mp8: TOPool;
function ExecuteInThread(const Proc: TThreadProc; Args: Pointer): TThread; overload;
function ExecuteInThread(const Proc: TThreadMeth; Args: Pointer): TThread; overload;
function ExecuteInThread(const Proc: TThreadNest; Args: Pointer): TThread; overload;

function GetSystemThreadCount: Integer;
{$IFDEF MSWINDOWS}
var
  SystemInfo: SYSTEM_INFO;
{$ELSE}

{$ENDIF}

implementation

function ExecuteInThread(const Proc: TThreadProc; Args: Pointer): TThread;
begin
  Result := TOThread.Create(True);
  with TOThread(Result) do
  begin
    FParams := Args;
    FThreadProc := Proc;
    Start
  end
end;

function ExecuteInThread(const Proc: TThreadMeth; Args: Pointer): TThread;
begin
  Result := TOThread.Create(True);
  with TOThread(Result) do
  begin
    FParams := Args;
    FMethod2 := Proc;
    Start
  end
end;

function ExecuteInThread(const Proc: TThreadNest; Args: Pointer): TThread;
begin
  Result := TOThread.Create(True);
  with TOThread(Result) do
  begin
    FParams := Args;
    FThreadNest := Proc;
    Start
  end
end;

function GetSystemThreadCount: Integer;
{$IFDEF MSWINDOWS}
begin
  GetSystemInfo(SystemInfo);
  Result := SystemInfo.DwNumberOfProcessors;
end;

{$ELSE}
begin

  Result := Max(TThread.ProcessorCount, 4);

end;
{$ENDIF}

{ TOPool }
constructor TOPool.Create(PoolCount: Integer);
var
  I: Integer;
begin
  inherited Create;
  OTCount := 0;
  // InitCriticalSection(FCriticalSection);
  if PoolCount = 0 then
    PoolCount := GetSystemThreadCount;
  SetLength(Pool, PoolCount);
  for I := 0 to High(Pool) do
  begin
    Pool[I] := TOThread.Create(False);
    Pool[I].FID := I;
    Pool[I].FPool := Self
  end;

  PoolDone := TEvent.Create();

  Id := GlobalId;
  Inc(GlobalId);
end;

class function TOPool.CPUName: TBuff;
{$IF defined(CPUX64)}
asm
  Push Rbx
  Mov  R8,    Result

  Mov Eax   , $80000002
  Cpuid
  Mov Dword Ptr [R8]   , Eax
  Mov Dword Ptr [R8+ 4]  , Ebx
  Mov Dword Ptr [R8+ 8]  , Ecx
  Mov Dword Ptr [R8+12]  , Edx

  Mov Eax   , $80000003
  Cpuid
  Mov Dword Ptr [R8+16]  , Eax
  Mov Dword Ptr [R8+20]  , Ebx
  Mov Dword Ptr [R8+24]  , Ecx
  Mov Dword Ptr [R8+28]  , Edx

  Mov Eax   , $80000004
  Cpuid
  Mov Dword Ptr [R8+32]  , Eax
  Mov Dword Ptr [R8+36]  , Ebx
  Mov Dword Ptr [R8+40]  , Ecx
  Mov Dword Ptr [R8+44]  , Edx
  Pop Rbx
end;

{$ELSE}
begin
  // assuming linux on ARM

end;
{$ENDIF}

class function TOPool.CPUFreqMhz: TCPUFreq;
{$IF defined(CPUX64)}
asm
  Push Rbx
  Mov  R8  , Result


  Mov        Eax, $15
  Cpuid                                    // read leaf 15h
  Cmp        Ecx  , 0                      // got any frequency ?
  Je         @Fr                           // no?, skip to reading leaf 16h
  Mov        TCPUFreq(R8).Xtal  , Ecx      // got crystal frequency in Hz

  Cvtsi2sd   Xmm0 , Eax                    // CPU clock Ratio D
  Cvtsi2sd   Xmm1 , Ebx                    // CPU clock ratio N
  Divsd      Xmm1 , Xmm0
  Movsd      TCPUFreq(R8).Ratio , Xmm1

  Mov        R9  , Rcx
  Imul       R9  , Rbx
  Xchg       R9  , Rax
  Idiv       R9
  Mov        R9  , Rax

@Fr:
  Mov        Eax, $16

  Cpuid

  Cmp        Ebx     , 0
  Je         @No
  Mov        TCPUFreq(R8).Base , Eax
  Mov        TCPUFreq(R8).Max  , Ebx
  Mov        TCPUFreq(R8).Bus  , Ecx
  Jmp        @Done

@No:
  shr        R9d               , 20   // divid by 1024
  Mov        TCPUFreq(R8).Max  , R9d

@Done:
  Pop Rbx
end;
{$ELSE}
begin

end;
{$ENDIF}

destructor TOPool.Destroy;
var
  I: Integer;
begin
  for I := High(Pool) downto 0 do
  begin
    Pool[I].Terminate;

{$IFDEF MSWINDOWS}
    SetEvent(Pool[I].Fire)
{$ELSE}
    Pool[I].Fire.SetEvent
{$ENDIF}
        ;

    // If terminated thread will be automatically destroyed when fired, no need to free
    // pool[i].free
    // freeandNil(Pool[i]);
  end;
  Dec(GlobalId);

  FreeAndNil(PoolDone);

  inherited Destroy;
end;

procedure TOPool.SetWorkers(Count: LongInt);
var
  I: LongInt;
begin
  for I := 0 to High(Pool) do
  begin
    Pool[I].Terminate;

{$IFDEF MSWINDOWS}
    ReSetEvent(Pool[I].Fire)
{$ELSE}
    Pool[I].Fire.ReSetEvent
{$ENDIF}
        ;

  end;
  SetLength(Pool, Count);
  for I := 0 to High(Pool) do
  begin
    Pool[I] := TOThread.Create(False);
    Pool[I].FID := I;
    Pool[I].FPool := Self
  end;
end;
{ TOThread }

procedure TOThread.Execute;
begin
  if Assigned(FThreadProc) then
  begin
    FThreadProc(FParams);
    FThreadProc := nil;
    Exit
  end;
  if Assigned(FMethod2) then
  begin
    FMethod2(FParams);
    FMethod2 := nil;
    Exit
  end;
  while not Terminated do
  begin

{$IFDEF MSWINDOWS}
    WaitForSingleObject(Fire, INFINITE)
{$ELSE}
    Fire.WaitFor()
{$ENDIF}
        ;

    if Terminated then
    begin
      // FBusy:=false;
      /// /RTLEventSetEvent(FPool.PoolDone);
      // InterLockedDecrement(FPool.OTCount);
      // {$else}
      /// /FPool.PoolDone.SetEvent;
      // TInterLocked.Decrement(FPool.OTCount);
      // {$endif}
      Exit;
    end;

    if Assigned(FProc) then
    begin
      FProc(FStart, FEnd, FParams);
      FProc := nil;
    end;

    if Assigned(FProcNested) then
    begin
      FProcNested(FStart, FEnd, FParams);
      FProcNested := nil;
    end;
    if Assigned(FMethod) then
    begin
      FMethod(FStart, FEnd, FParams);
      FMethod := nil;
    end;

    if Assigned(FThreadProc2) then
    begin
      while FStart < FEnd do
      begin
        FThreadProc2(FStart, FParams);
        if Assigned(FSync) then
          Queue(FSync);
        Inc(FStart, FSpan);
      end;
      FThreadProc2 := nil;
    end;
    if Assigned(FThreadProcNested) then
    begin
      while FStart < FEnd do
      begin
        FThreadProcNested(FStart, FParams);
        if Assigned(FSync) then
          Queue(FSync);
        Inc(FStart, FSpan);
      end;
      FThreadProcNested := nil;
    end;

{$IFDEF MSWINDOWS}
    //        ResetEvent(Fire);
{$ELSE}
    Fire.ReSetEvent();
{$ENDIF}

    FBusy := False;

    FPool.PoolDone.SetEvent;
    TInterLocked.Decrement(FPool.OTCount);

  end
end;

constructor TOThread.Create(CreateSuspended: Boolean; const StackSize: NativeUInt);
begin
  inherited Create(CreateSuspended);
  FreeOnTerminate := True;;

{$IFDEF MSWINDOWS}
  Fire := CreateEvent(nil, False, False, nil);
{$ELSE}
  Fire := TEvent.Create();
{$ENDIF}

  //  Priority:=tpHighest;
  FBusy := False;
end;

constructor TOThread.Create(const Proc: TGroupProc; const Params: Pointer);
begin
  inherited Create(True);
  FreeOnTerminate := True;;
  FProc := Proc;
  FParams := Params
end;

destructor TOThread.Destroy;
begin

{$IFDEF MSWINDOWS}
  CloseHandle(Fire);
{$ELSE}
  Fire.Free;
{$ENDIF}

{$IFDEF DEBUG}
  // writeln('[Steroids] Freeing thread pool, Current pool size is : ', length(FPool.Pool));
{$ENDIF}
  inherited Destroy;
end;

function TOPool.IsBusy: Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 0 to Length(Pool) - 1 do
    if Pool[I].FBusy then
      Exit(True)
end;

{$IFDEF POSIX}
procedure TOPool.SetPriority(const Priority: Integer);
{$ELSE}
procedure TOPool.SetPriority(const Priority: TGroupPriority);
{$ENDIF}
var
  I: Integer;
begin
  for I := 0 to Length(Pool) - 1 do
    Pool[I].Priority := Priority;
end;

procedure TOPool.WaitForPool;
begin
  while IsBusy() do

    PoolDone.WaitFor()

end;

function TOPool.GetWorkers: LongInt;
begin
  Result := Length(Pool)
end;

procedure TOPool.&for(const Proc: TGroupProc; const LocalFrom, LocalTo: IntPtr; const Params: Pointer);
var
  I, N, GroupM, GroupT, Ii: IntPtr;
begin
  if LocalTo < LocalFrom then
    Exit;
  WaitForPool;
  // while OTCount>0 do;// the pool is still hot! wait for it to cooldown before jumping in.
  N := LocalTo - LocalFrom;
  Ii := 0;
  GroupT := Ceil((N + 1) / Length(Pool));
  GroupM := N mod Length(Pool);
  if GroupT > 0 then
    for I := 0 to High(Pool) do
    begin
      Pool[I].FStart := Ii;
      Inc(Ii, GroupT - LongInt(I > GroupM));
      Pool[I].FEnd := Ii - 1;
      if Ii <= Pool[I].FStart then
        Break;
      Pool[I].FParams := Params;
      Pool[I].FProc := Proc;
      Pool[I].FBusy := True;
      // Pool[i].FSync:=sync;

      TInterLocked.Increment(OTCount);
{$IFDEF MSWINDOWS}
      SetEvent(Pool[I].Fire);
{$ELSE}
      Pool[I].Fire.SetEvent();
{$ENDIF}

    end;
  WaitForPool
end;

procedure TOPool.&for(const Proc: TGroupProcNested; const LocalFrom, LocalTo: IntPtr; const Params: Pointer);
var
  I, N, GroupM, GroupT, Ii: IntPtr;
begin
  if LocalTo < LocalFrom then
    Exit;
  WaitForPool;
  // while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
  N := LocalTo - LocalFrom;
  Ii := 0;
  GroupT := Ceil((N + 1) / Length(Pool));
  GroupM := N mod Length(Pool);
  if GroupT > 0 then
    for I := 0 to High(Pool) do
    begin
      Pool[I].FStart := Ii;
      Inc(Ii, GroupT - LongInt(I > GroupM));
      Pool[I].FEnd := Ii - 1;
      if Ii <= Pool[I].FStart then
        Break;
      Pool[I].FParams := Params;
      Pool[I].FProcNested := Proc;
      Pool[I].FBusy := True;
      // Pool[i].FSync:=sync;

      TInterLocked.Increment(OTCount);
{$IFDEF MSWINDOWS}
      SetEvent(Pool[I].Fire);
{$ELSE}
      Pool[I].Fire.SetEvent();
{$ENDIF}

    end;

  WaitForPool
end;

procedure TOPool.&for(const Proc: TGroupMethod; const LocalFrom, LocalTo: IntPtr; const Params: Pointer);
var
  I, N, GroupM, GroupT, Ii: IntPtr;
begin
  if LocalTo < LocalFrom then
    Exit;
  WaitForPool;
  // while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
  N := LocalTo - LocalFrom;
  Ii := 0;
  GroupT := Ceil((N + 1) / Length(Pool));
  GroupM := N mod Length(Pool);
  if GroupT > 0 then
    for I := 0 to High(Pool) do
    begin
      Pool[I].FStart := Ii;
      Inc(Ii, GroupT - LongInt(I > GroupM));
      Pool[I].FEnd := Ii - 1;
      if Ii <= Pool[I].FStart then
        Break;
      Pool[I].FParams := Params;
      Pool[I].FMethod := Proc;
      Pool[I].FBusy := True;
      // Pool[i].FSync:=sync;

      TInterLocked.Increment(OTCount);
{$IFDEF MSWINDOWS}
      SetEvent(Pool[I].Fire);
{$ELSE}
      Pool[I].Fire.SetEvent();
{$ENDIF}

    end;
  WaitForPool
end;

procedure TOPool.&for(
    const Proc: TThreadProc2;
    const Start, Count: IntPtr;
    const Args: Pointer;
    const Step: IntPtr;
    const Sync: TThreadMethod
);
var
  Span, I, Ii: IntPtr;
begin
  // if count < start then exit;
  WaitForPool;
  // while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
  Span := Step * Length(Pool);
  Ii := Start;
  for I := 0 to High(Pool) do
  begin
    Pool[I].FStart := Ii;
    Inc(Ii, Step);
    if Ii > Count then
      Break;
    Pool[I].FSpan := Span;
    Pool[I].FEnd := Count;
    Pool[I].FParams := Args;
    Pool[I].FThreadProc2 := Proc;
    Pool[I].FBusy := True;
    Pool[I].FSync := Sync;

    TInterLocked.Increment(OTCount);
{$IFDEF MSWINDOWS}
    SetEvent(Pool[I].Fire);
{$ELSE}
    Pool[I].Fire.SetEvent();
{$ENDIF}

  end;
  WaitForPool
end;

procedure TOPool.&for(
    const Proc: TThreadProcNested;
    const Start, Count: IntPtr;
    const Args: Pointer;
    const Step: IntPtr;
    const Sync: TThreadMethod
);
var
  Span, I, Ii: IntPtr;
begin
  // if count < start then exit;
  WaitForPool;
  // while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
  Span := Step * Length(Pool);
  Ii := Start;
  for I := 0 to High(Pool) do
  begin
    Pool[I].FStart := Ii;
    Inc(Ii, Step);
    if Ii > Count then
      Break;
    Pool[I].FSpan := Span;
    Pool[I].FEnd := Count;
    Pool[I].FParams := Args;
    Pool[I].FThreadProcNested := Proc;
    Pool[I].FBusy := True;
    Pool[I].FSync := Sync;

    TInterLocked.Increment(OTCount);
{$IFDEF MSWINDOWS}
    SetEvent(Pool[I].Fire);
{$ELSE}
    Pool[I].Fire.SetEvent();
{$ENDIF}

  end;
  WaitForPool
end;

initialization
  Mp := TOPool.Create;
  Mp2 := TOPool.Create(2);
  Mp4 := TOPool.Create(4);
  Mp8 := TOPool.Create(8);
  //

finalization
  if Assigned(Mp8) then
    FreeAndNil(Mp8);
  if Assigned(Mp4) then
    FreeAndNil(Mp4);
  if Assigned(Mp2) then
    FreeAndNil(Mp2);
  if Assigned(Mp) then
    FreeAndNil(Mp);
end.
