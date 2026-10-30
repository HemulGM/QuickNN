// {$if defined(CPUX64) and defined(USE_AVX2)}
{$IF defined(CPUX64)}

(*
  carful with windows and linux calling conventions, read :
  https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170
*)

const
  SIMD_REGS = 8;
  // ONE : single =1.0;

  {$IF SIMD_REGS = 8}
  SIMD_SHFT =
      3
  {$ELSE}
  SIMD_SHFT =
      2
  {$ENDIF}
      ;

  SIMD_OFF = SIMD_REGS * SizeOf(Single);
  YMMD: array[0..7] of Int32 = (0, 1, 2, 3, 4, 5, 6, 7);

const
  L2_E: Single = 1.4426950408889634; // log2(e);
  C2: Single = 1.00172476;
  C1: Single = 0.657636276;
  C0: Single = 0.3371894346;
  // A   :single = 12102203.0; (* (1 << 23) / log(2) *)
  // M   :longint = $ff800000; (* mask for integer bits *)
  // ttm23 :single = 1.1920929e-7; (* exp2(-23) *)

  // MAX_EXP =  8.8722839052068352E+001;
  // MIN_EXP = -8.7336544750553102E+001;

  MAX_EXP = 8.87E+001;
  MIN_EXP = -8.73E+001;

  ONE: array[0..7] of Single = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  ZERO: array[0..7] of Single = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  MX: array[0..7] of Single = (MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP);
  MN: array[0..7] of Single = (MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP);

  S1442 = 1.4426950408889634; // log2(e)
  S0693 = -0.6931471805599453;
  S0008 = 0.00833333;
  S0041 = 0.04166667;
  S0166 = 0.16666667;
  S0500 = 0.5;

  P1442: array[0..7] of Single = (S1442, S1442, S1442, S1442, S1442, S1442, S1442, S1442);
  P0693: array[0..7] of Single = (S0693, S0693, S0693, S0693, S0693, S0693, S0693, S0693);
  P0008: array[0..7] of Single = (S0008, S0008, S0008, S0008, S0008, S0008, S0008, S0008);
  P0041: array[0..7] of Single = (S0041, S0041, S0041, S0041, S0041, S0041, S0041, S0041);
  P0166: array[0..7] of Single = (S0166, S0166, S0166, S0166, S0166, S0166, S0166, S0166);
  P0500: array[0..7] of Single = (S0500, S0500, S0500, S0500, S0500, S0500, S0500, S0500);

procedure SExpAVX2V2(const N: NativeInt; const Src: System.PSingle; const Dst: System.PSingle); assembler;
asm

//if (x < -87.33) then  exit( 0.0);
//if (x > 88.72)  then exit( 1e38);
  //result := exp(x)
    //if (x < -87.33) then  exit( 0.0);
    //if (x > 88.72)  then exit( 1e38);
    ////if (x < MIN_EXP) then  x := MIN_EXP;
    ////if (x > MAX_EXP)  then  x := MAX_EXP;
    //n := floor(x * 1.4426950408889634 + 0.5);
    //r := x - n * 0.6931471805599453;
    //p := 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 +
    //          r * (0.04166667 + r * 0.00833333))));
    //v.f := p;
    //inc(v.i, trunc(n) shl 23);
    //result := v.f;
  Sub                  Rsp      , $10*2                     // making stack space to save one xmm size register
  Vmovdqu              [Rsp+$00], Xmm6
  Vmovdqu              [Rsp+$10], Xmm7

  Mov           R11   , N
  shr           R11   , 3
  Jz            @Rem

@while:
//n := floor(x * 1.4426950408889634 + 0.5);

  Vmovups       Ymm3  , Yword[Src]
  Vcmpgeps      Ymm6  , Ymm3        , [Rip + MX]
  Vcmpleps      Ymm7  , Ymm3        , [Rip + MN]
  Vblendvps     Ymm3  , Ymm3 , [Rip + MX], Ymm6
  Vblendvps     Ymm3  , Ymm3 , [Rip + MN], Ymm7


  Vmovaps       Ymm2  , Ymm3
  Vmulps        Ymm2  , Ymm2     , [Rip + P1442]
  Vaddps        Ymm2  , Ymm2     , [Rip + P0500]            // X L2E + 0.5
  Vroundps      Ymm2  , Ymm2     , 9          // 9 means floor()  // ymm2 is n

//r := x  + n * (-0.6931471805599453);
  Vmovaps       Ymm1  , Ymm3
  Vfmadd231ps   Ymm1  , Ymm2     , [Rip + P0693] // ymm1 is now r

  Vmovaps       Ymm0  , Ymm1                  // p := r

//p := p * 0.00833333 + 0.04166667
  Vmovups       Ymm5  , [Rip +P0008]
  Vfmadd213ps   Ymm0  , Ymm5     , [Rip +P0041]

//p := r*p + 0.16666667
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0166]

//p := r*p + 0.5
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0500]

//p := r*p + 1.0
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

//p := r*p + 1.0
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

// all the above performs now :
//   p := 1.0 + r*(1.0 + r*(0.5 + r*(0.16666667 + r*(0.04166667 + r*0.00833333)))); // rank 5 approximation

//v.f := p;
//inc(v.i, trunc(n) shl 23);
  Vcvtps2dq     Ymm2  , Ymm2
  Vpslld        Ymm2  , Ymm2     , 23
  Vpaddd        Ymm0  , Ymm0     , Ymm2
//result := v.f;
  Vmovups       Yword [Dst]      , Ymm0

  Add           Src   , 32
  Add           Dst   , 32
  Dec           R11
  Jnz           @while

@Rem:
  and           N   , 7
  Jz            @Done
  Mov           Rax  , N
  Vmovd         Xmm3 , Eax
  Vpbroadcastd  Ymm3 , Xmm3
  Vpcmpgtd      Ymm3 , Ymm3, YMMD[Rip]
  Vcmpgeps      Ymm6 , Ymm3        , [Rip + MX]
  Vcmpleps      Ymm7 , Ymm3        , [Rip + MN]
  Vblendvps     Ymm3 , Ymm3 , [Rip + MX], Ymm6
  Vblendvps     Ymm3 , Ymm3 , [Rip + MN], Ymm7

//n := floor(x * 1.4426950408889634 + 0.5);
  Vmaskmovps    Ymm2  , Ymm3     , Yword[Src]
  Vmulps        Ymm2  , Ymm2     , [Rip +P1442]
  Vaddps        Ymm2  , Ymm2     , [Rip +P0500]            // X L2E + 0.5
  Vroundps      Ymm2  , Ymm2     , 9          // 9 means floor()  // ymm2 is n

//r := x  + n * (-0.6931471805599453);
  Vmaskmovps    Ymm1  , Ymm3     , Yword[Src]
  Vfmadd231ps   Ymm1  , Ymm2     , [Rip +P0693] // ymm1 is now r


  Vmovaps       Ymm0  , Ymm1     // p := r

//p := p * 0.00833333 + 0.04166667
  Vmovups       Ymm5  , [Rip +P0008]
  Vfmadd213ps   Ymm0  , Ymm5     , [Rip +P0041]

//p := r*p + 0.16666667
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0166]

//p := r*p + 0.5
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0500]

//p := r*p + 1.0
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

//p := r*p + 1.0
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

// all the above performs now :
//   p := 1.0 + r*(1.0 + r*(0.5 + r*(0.16666667 + r*(0.04166667 + r*0.00833333)))); // rank 5 approximation

//v.f := p;
//inc(v.i, trunc(n) shl 23);
  Vcvtps2dq     Ymm2  , Ymm2
  Vpslld        Ymm2  , Ymm2     , 23
  Vpaddd        Ymm0  , Ymm0     , Ymm2
//result := v.f;
  Vmaskmovps    Yword [Dst] , Ymm3 , Ymm0

@Done:

  Vmovdqu       Xmm6  , [Rsp+$00]
  Vmovdqu       Xmm7  , [Rsp+$10]
  Add           Rsp   , $10*2                     // making stack space to save one xmm size register

end;

procedure SExpAVX2(const N: NativeInt; const Src: System.PSingle; const Dst: System.PSingle); assembler;
asm
  Sub                  Rsp      , $10*2                     // making stack space to save one xmm size register
  Vmovdqu              [Rsp+$00], Xmm6
  Vmovdqu              [Rsp+$10], Xmm7

  Vbroadcastss  Ymm3  , [Rip + L2_E]
  Vbroadcastss  Ymm4  , [Rip + C1]
  Vbroadcastss  Ymm5  , [Rip + C2]

  Mov           R11   , N
  shr           R11   , 3
  Jz            @Rem

@while:
  //vxorps        ymm0  , ymm0        , ymm0              // zero
  Vmovups       Ymm1  , Yword [Src]
  Vcmpgeps      Ymm6  , Ymm1        , [Rip + MX]
  Vcmpleps      Ymm7  , Ymm1        , [Rip + MN]
  Vblendvps     Ymm1  , Ymm1 , [Rip + MX], Ymm6
  Vblendvps     Ymm1  , Ymm1 , [Rip + MN], Ymm7
  Vmulps        Ymm1  , Ymm3        , Ymm1
  Vroundps      Ymm2  , Ymm1        , 9
  Vsubps        Ymm1  , Ymm1        , Ymm2
  Vcvtps2dq     Ymm0  , Ymm2
  Vbroadcastss  Ymm2  , [Rip + C0]
  Vfmadd213ps   Ymm2  , Ymm1        , Ymm4
  Vpslld        Ymm0  , Ymm0        , 23
  Vfmadd213ps   Ymm1  , Ymm2        , Ymm5
  Vpaddd        Ymm0  , Ymm0        , Ymm1
  Vmovups       Yword [Dst] , Ymm0

  Add           Src   , 32
  Add           Dst   , 32
  Dec           R11
  Jnz           @while
  Vzeroupper
  and           N   , 7
  Jz            @Done

@Rem:

  Vmovss        Xmm1  , Dword [Src]
  Vcmpgeps      Xmm6  , Xmm1        , [Rip + MX]
  Vcmpleps      Xmm7  , Xmm1        , [Rip + MN]
  Vblendvps     Xmm1  , Xmm1 , [Rip + MX], Xmm6
  Vblendvps     Xmm1  , Xmm1 , [Rip + MN], Xmm7
  Vmulss        Xmm1  , Xmm3        , Xmm1
  Roundss       Xmm2  , Xmm1        , 9
  Vsubss        Xmm1  , Xmm1        , Xmm2
  Vcvtps2dq     Xmm0  , Xmm2
  Vmovss        Xmm2  , [Rip + C0]
  Vfmadd213ss   Xmm2  , Xmm1        , Xmm4
  Vpslld        Xmm0  , Xmm0        , 23
  Vfmadd213ss   Xmm1  , Xmm2        , Xmm5
  Vpaddd        Xmm0  , Xmm0       , Xmm1
  Vmovss        Dword [Dst] , Xmm0

  Add           Src   , 4
  Add           Dst   , 4
  Dec           N
  Jnz           @Rem

@Done:
  Vmovdqu              Xmm6     , [Rsp+$00]
  Vmovdqu              Xmm7     , [Rsp+$10]
  Add                  Rsp      , $10*2                     // restoring stack
end;

// AKA SWISH
// sigmoid or dst can be nil

procedure SSiLUAVX2(
    const N: NativeInt;
    const Src, Up, OutSigmoid, Dst: System.PSingle
); assembler; // a function that does sigmoid, gated and SiLU
asm
  Mov                  Rax      , Dst
  Sub                  Rsp      , $10*2                     // making stack space to save one xmm size register
  Vmovdqu              [Rsp+$00], Xmm6
  Vmovdqu              [Rsp+$10], Xmm7

  Vbroadcastss  Ymm3  , [Rip + L2_E]
  Vbroadcastss  Ymm4  , [Rip + C1]
  Vbroadcastss  Ymm5  , [Rip + C2]

  Mov           R11   , N
  shr           R11   , 3
  Jz            @Rem

@while:
  Vxorps        Ymm0  , Ymm0        , Ymm0              // zero
  Vsubps        Ymm1  , Ymm0        , Yword [Src]       // t1 = -src

  Vcmpgeps      Ymm6  , Ymm1        , [Rip + MX]        // clamp t1 to [MIN_EXP, MAX_EXP]
  Vcmpleps      Ymm7  , Ymm1        , [Rip + MN]
  Vblendvps     Ymm1  , Ymm1 , [Rip + MX], Ymm6
  Vblendvps     Ymm1  , Ymm1 , [Rip + MN], Ymm7

  Vmulps        Ymm1  , Ymm3        , Ymm1              // t1 = t1 * log2(e)
  Vroundps      Ymm2  , Ymm1        , 9                 // t2 = floor(t1 * log2(e))
  Vsubps        Ymm1  , Ymm1        , Ymm2              // t1 = t1 - t2
  Vcvtps2dq     Ymm0  , Ymm2                            // t0 = cast to integer(t2)
  Vbroadcastss  Ymm2  , [Rip + C0]
  Vfmadd213ps   Ymm2  , Ymm1        , Ymm4              // t2 = C0 * t1 + C1
  Vpslld        Ymm0  , Ymm0        , 23                // t0 = t0 shl 23
  Vfmadd213ps   Ymm1  , Ymm2        , Ymm5              // t1 = t1 * t2 + C2
  Vpaddd        Ymm0  , Ymm0        , Ymm1              // t0 = t0 + integer(t1) => exp(t1)
  Vaddps        Ymm1  , Ymm0        , [Rip + ONE]       // 1 +exp(-src)
  Vrcpps        Ymm1  , Ymm1                            // 1/(1+exp(-src))

  Cmp           OutSigmoid   , 0
  Je            @Skip1
  Vmovups       Yword [OutSigmoid] , Ymm1
  Add           OutSigmoid   , 32
@Skip1:
  Cmp           Rax       , 0
  Je            @Skipup1
  Vmulps        Ymm1            , Ymm1 ,   Yword [Src]
  Cmp           Up        , 0
  Je            @Skipdst1
  Vmulps        Ymm1            , Ymm1 ,   Yword [Up]
  Add           Up              , 32
@Skipdst1:
  Vmovups       Yword [Rax]     , Ymm1
  Add           Rax             , 32

@Skipup1:
  Add           Src       , 32
  Dec           R11
  Jnz           @while

  Vzeroupper
  and           N   , 7
  Jz            @Done

@Rem:

  Vpxor         Xmm0  , Xmm0        , Xmm0
  Vsubss        Xmm1  , Xmm0        , Dword [Src]              //-src
  Vcmpgeps      Xmm6  , Xmm1        , [Rip + MX]
  Vcmpleps      Xmm7  , Xmm1        , [Rip + MN]
  Vblendvps     Xmm1  , Xmm1 , [Rip + MX], Xmm6
  Vblendvps     Xmm1  , Xmm1 , [Rip + MN], Xmm7
  Vmulss        Xmm1  , Xmm3        , Xmm1
  Roundss       Xmm2  , Xmm1        , 9
  Vsubss        Xmm1  , Xmm1        , Xmm2
  Vcvtps2dq     Xmm0  , Xmm2
  Vmovss        Xmm2  , [Rip + C0]
  Vfmadd213ss   Xmm2  , Xmm1        , Xmm4
  Vpslld        Xmm0  , Xmm0        , 23
  Vfmadd213ss   Xmm1  , Xmm2        , Xmm5
  Vpaddd        Xmm0  , Xmm0        , Xmm1
  Vaddss        Xmm1  , Xmm0        , [Rip + ONE]       // 1 +exp(-src)
  Vrcpss        Xmm1  , Xmm1        , Xmm1              // 1/(1+exp(-src))
  //vpcmpeqd      xmm0  , xmm0        , xmm0              // set ymm0 to to 0xffffffff
  //vpandn        xmm6  , xmm6        , xmm0              // ymm6 := not ymm6
  //vpandn        xmm7  , xmm7        , xmm0              // ymm6 := not ymm6
  //vblendvps     xmm1  , xmm1  , [rip+one] , xmm6
  //vblendvps     xmm1  , xmm1  , [rip+zero]  , xmm7
  Cmp           OutSigmoid   , 0   // sigmoid is NULL
  Je            @Skip2
  Vmovss        Dword [OutSigmoid] , Xmm1
  Add           OutSigmoid         , 4
@Skip2:
  Cmp           Rax       , 0
  Je            @Skipup2
  Vmulss        Xmm1            , Xmm1 ,   Dword [Src]
  Cmp           Up        , 0
  Je            @Skipdst2
  Vmulss        Xmm1            , Xmm1 ,   Dword [Up]
  Add           Up              , 4
@Skipdst2:
  Vmovss        Dword [Rax]     , Xmm1
  Add           Rax             , 4

@Skipup2:
  Add           Src             , 4
  Dec           N
  Jnz           @Rem
@Done:
  Vmovdqu       Xmm6            , [Rsp+$00]
  Vmovdqu       Xmm7            , [Rsp+$10]
  Add           Rsp             , $10*2                     // restoring stack
end;

// const
//  YMMD : array [0..7] of longword =(0,1,2,3,4,5,6,7);//: array[0..7] of single = ()
function SMaxAVX2(const N: NativeInt; const Src: System.PSingle): Single; assembler;
asm
  Vbroadcastss  Ymm0 , Dword [Src]

  Mov           R8   , N
  shr           R8   , 5
  Jz            @Rem1

  Vmovaps       Ymm1 , Ymm0
  Vmovaps       Ymm2 , Ymm0
  Vmovaps       Ymm3 , Ymm0

@While1:
  Vmaxps        Ymm0 , Ymm0, Yword [Src]
  Vmaxps        Ymm1 , Ymm1, Yword [Src +32  ]
  Vmaxps        Ymm2 , Ymm2, Yword [Src +32*2]
  Vmaxps        Ymm3 , Ymm3, Yword [Src +32*3]
  Add           Src  , 32*4
  Dec           R8
  Jnz           @While1

  Vmaxps        Ymm0, Ymm0, Ymm1
  Vmaxps        Ymm2, Ymm2, Ymm3
  Vmaxps        Ymm0, Ymm0, Ymm2

@Rem1:

  Mov           R8   , N
  and           R8   , 31
  shr           R8   , 3
  Jz            @Rem

@while:
  Vmaxps        Ymm0 , Ymm0, Yword [Src]
  Add           Src  , 32
  Dec           R8
  Jnz           @while

@Rem:
  Mov           R8   , N
  and           R8   , 7
  Jz            @Done

  Vmovd         Xmm1 , R8d
  Vpbroadcastd  Ymm1 , Xmm1
  Vpcmpgtd      Ymm1 , Ymm1, YMMD[Rip]
  //      vmaskmovps   (%rsi,%rax,sz), %ymm1, %ymm2
  Vblendvps     Ymm1 , Ymm0, Yword[Src], Ymm1
  Vmaxps        Ymm0 , Ymm0, Ymm1

@Done:
  Vextractf128  Xmm1 , Ymm0, 1
  Vzeroupper
  Maxps         Xmm0 , Xmm1
  Movhlps       Xmm1 , Xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  Maxps         Xmm0 , Xmm1
  Movshdup      Xmm1 , Xmm0
  Maxss         Xmm0 , Xmm1

end;

function SMinAVX2(const N: NativeInt; const Src: System.PSingle): Single; assembler;
asm
  Vbroadcastss  Ymm0 , Dword [Src]

  Mov           R8   , N
  shr           R8   , 5
  Jz            @Rem1

  Vmovaps       Ymm1 , Ymm0
  Vmovaps       Ymm2 , Ymm0
  Vmovaps       Ymm3 , Ymm0

@While1:
  Vminps        Ymm0 , Ymm0, Yword [Src]
  Vminps        Ymm1 , Ymm1, Yword [Src +32  ]
  Vminps        Ymm2 , Ymm2, Yword [Src +32*2]
  Vminps        Ymm3 , Ymm3, Yword [Src +32*3]
  Add           Src  , 32*4
  Dec           R8
  Jnz           @While1

  Vminps        Ymm0, Ymm0, Ymm1
  Vminps        Ymm2, Ymm2, Ymm3
  Vminps        Ymm0, Ymm0, Ymm2

@Rem1:

  Mov           R8   , N
  and           R8   , 31
  shr           R8   , 3
  Jz            @Rem

@while:
  Vminps        Ymm0 , Ymm0, Yword [Src]
  Add           Src  , 32
  Dec           R8
  Jnz           @while

@Rem:
  Mov           R8   , N
  and           R8   , 7
  Jz            @Done

  Vmovd         Xmm1 , R8d
  Vpbroadcastd  Ymm1 , Xmm1
  Vpcmpgtd      Ymm1 , Ymm1, YMMD[Rip]
  //      vmaskmovps   (%rsi,%rax,sz), %ymm1, %ymm2
  Vblendvps     Ymm1 , Ymm0, Yword[Src], Ymm1
  Vminps        Ymm0 , Ymm0, Ymm1

@Done:
  Vextractf128  Xmm1 , Ymm0, 1
  Vzeroupper
  Minps         Xmm0 , Xmm1
  Movhlps       Xmm1 , Xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  Minps         Xmm0 , Xmm1
  Movshdup      Xmm1 , Xmm0
  Minss         Xmm0 , Xmm1

end;

procedure SScaleInplaceAVX2(const N: NativeInt; const Src: System.PSingle; const Scale: Single); assembler;
asm
  Vbroadcastss  Ymm4 , Scale
  Mov           Rax  , N
  shr           Rax  , 5
  Jz            @Rem1

@while:
  Vmulps        Ymm0 , Ymm4, Yword[Src]
  Vmulps        Ymm1 , Ymm4, Yword[Src+32*1]
  Vmulps        Ymm2 , Ymm4, Yword[Src+32*2]
  Vmulps        Ymm3 , Ymm4, Yword[Src+32*3]

  Vmovups       Yword[Src]      , Ymm0
  Vmovups       Yword[Src+32*1] , Ymm1
  Vmovups       Yword[Src+32*2] , Ymm2
  Vmovups       Yword[Src+32*3] , Ymm3

  Add           Src  , 32*4
  Dec           Rax
  Jnz           @while

@Rem1:
  Mov           Rax  , N
  and           Rax  , 31
  shr           Rax  , 3
  Jz            @Rem

@While1:
  Vmulps        Ymm0 , Ymm4, Yword[Src]
  Vmovups       Yword[Src] , Ymm0

  Add           Src  , 32
  Dec           Rax
  Jnz           @While1

@Rem:
  Mov           Rax  , N
  and           Rax  , 7
  Jz            @Done
  Vmovd         Xmm1 , Eax
  Vpbroadcastd  Ymm1 , Xmm1
  Vpcmpgtd      Ymm1 , Ymm1, YMMD[Rip]
  Vmaskmovps    Ymm0 , Ymm1, Yword[Src]
  Vmulps        Ymm0,  Ymm0, Ymm4
  Vmaskmovps    Yword[Src] , Ymm1, Ymm0
@Done:
  Vzeroupper
end;

procedure SSoftmaxAVX2(const N: NativeInt; const Src: System.PSingle); assembler;
asm
//exp
  //n := floor(x * 1.4426950408889634 + 0.5);
  //r := x - n * 0.6931471805599453;
  //p := 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.04166667 + r * 0.00833333))));
  //v.f := p;
  //inc(v.i, trunc(n) shl 23);
  //result := v.f;

//softmax
  //max_val := QNNMax(N, x);
  //sum := 0.0;
  //for i := 0 to N-1 do begin
  //    x[i] := fast_exp(x[i]-max_val);
  //    //x[i] := exp(x[i]-max_val);
  //    sum := sum + x[i]
  //end;
  //inv_sum := single(1.0) / sum;
  //QNNScaleInplace(x, inv_sum, N);
  Push          Src
  Push          N
  CALL SMaxAVX2// max(src)
  Pop           N
  Pop           Src

  Sub                  Rsp      , $10*4                     // making stack space to save one xmm size register
  Vmovdqu              [Rsp+$00], Xmm6
  Vmovdqu              [Rsp+$10], Xmm7
  Vmovdqu              [Rsp+$20], Xmm8
  Vmovdqu              [Rsp+$30], Xmm9

  Mov           R8    , Src
(*old approximation
  //vbroadcastss  ymm3  , [rip + L2E]
  //vbroadcastss  ymm4  , [rip + C1]
  //vbroadcastss  ymm5  , [rip + C2]
*)
  Vbroadcastss  Ymm6, Xmm0                       // = max_src
  Vxorps        Ymm7, Ymm7, Ymm7                 // = sum = 0
  Mov  Rax , N
  shr  Rax , 3
  Jz   @Rem

@while:
  Vmovups       Ymm3  , Yword[Src]
  Vsubps        Ymm3  , Ymm3 , Ymm6             // t1 = src - max_src

  Vcmpgeps      Ymm8  , Ymm3        , [Rip + MX]
  Vcmpleps      Ymm9  , Ymm3        , [Rip + MN]
  Vblendvps     Ymm3  , Ymm3 , [Rip + MX], Ymm8
  Vblendvps     Ymm3  , Ymm3 , [Rip + MN], Ymm9
  //vcmpgeps      ymm8  , ymm3        , [rip + MX]        // clamp t1 to [MIN_EXP, MAX_EXP]
  //vcmpleps      ymm9  , ymm3        , [rip + MN]
  //vblendvps     ymm3  , ymm3 , [rip + MX], ymm8
  //vblendvps     ymm3  , ymm3 , [rip + MN], ymm9

(* old approximation)
  //vmulps        ymm1  , ymm3    , ymm1      // t1 = t1 * log2(e)
  //vroundps      ymm2  , ymm1    , 9         // t2 = floor(t1 * log2(e))
  //vsubps        ymm1  , ymm1    , ymm2      // t1 = t1 - t2
  //vcvtps2dq     ymm0  , ymm2                // t0 = trunc(t2)
  //vbroadcastss  ymm2  , [rip + C0]
  //vfmadd213ps   ymm2  , ymm1    , ymm4      // t2 = C0 * t1 + C1
  //vpslld        ymm0  , ymm0    , 23        // t0 = t0 shl 23
  //vfmadd213ps   ymm1  , ymm2    , ymm5      // t1 = t1 * t2 + C2
  //vpaddd        ymm0  , ymm0    , ymm1      // t0 = t0 + integer(t1) => exp(t1)
  //
  //vblendvps     ymm0  , ymm0 , [rip + ZERO], ymm9
  //vmovups       yword[src]   , ymm0
*)

  Vmulps        Ymm2  , Ymm3     , [Rip +P1442]
  Vaddps        Ymm2  , Ymm2     , [Rip +P0500]            // X L2E + 0.5
  Vroundps      Ymm2  , Ymm2     , 9          // 9 means floor()  // ymm2 is n

//r := x  + n * (-0.6931471805599453);
  Vmovaps       Ymm1  , Ymm3
  Vfmadd231ps   Ymm1  , Ymm2     , [Rip +P0693] // ymm1 is now r

  Vmovaps       Ymm0  , Ymm1                  // p := r

//p := p * 0.00833333 + 0.04166667
  Vmovups       Ymm5  , [Rip +P0008]
  Vfmadd213ps   Ymm0  , Ymm5     , [Rip +P0041]

//p := r*p + 0.16666667
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0166]

//p := r*p + 0.5
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0500]

//p := r*p + 1.0
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

//p := r*p + 1.0
  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

// all the above performs now :
//   p := 1.0 + r*(1.0 + r*(0.5 + r*(0.16666667 + r*(0.04166667 + r*0.00833333)))); // rank 5 approximation

//v.f := p;
//inc(v.i, trunc(n) shl 23);
  Vcvtps2dq     Ymm2  , Ymm2
  Vpslld        Ymm2  , Ymm2    , 23
  Vpaddd        Ymm0  , Ymm0    , Ymm2
//result := v.f;
  Vmovups       Yword [Src]     , Ymm0
  Vaddps        Ymm7  , Ymm7    , Ymm0


  Add           Src   , 32
  Dec           Rax
  Jnz           @while


@Rem:
  Mov           Rax , N
  and           Rax , 7
  Jz            @div
  Vmovd         Xmm4 , Eax
  Vpbroadcastd  Ymm4 , Xmm4
  Vpcmpgtd      Ymm4 , Ymm4, YMMD[Rip]
  Vmaskmovps    Ymm3 , Ymm4, Yword[Src]

  Vsubps        Ymm3  , Ymm3 , Ymm6         // t1 = src - max_src

  Vcmpgeps      Ymm8  , Ymm3        , [Rip + MX]   // clip to [-87.3 -88.7]
  Vcmpleps      Ymm9  , Ymm3        , [Rip + MN]
  Vblendvps     Ymm3  , Ymm3 , [Rip + MX], Ymm8
  Vblendvps     Ymm3  , Ymm3 , [Rip + MN], Ymm9

  Vmulps        Ymm2  , Ymm3     , [Rip +P1442]
  Vaddps        Ymm2  , Ymm2     , [Rip +P0500]            // X L2E + 0.5
  Vroundps      Ymm2  , Ymm2     , 9          // 9 means floor()  // ymm2 is n

  Vmovaps       Ymm1  , Ymm3
  Vfmadd231ps   Ymm1  , Ymm2     , [Rip +P0693] // ymm1 is now r

  Vmovaps       Ymm0  , Ymm1                  // p := r

  Vmovups       Ymm5  , [Rip +P0008]
  Vfmadd213ps   Ymm0  , Ymm5     , [Rip +P0041]

  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0166]

  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +P0500]

  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

  Vfmadd213ps   Ymm0  , Ymm1     , [Rip +ONE]

// all the above performs now :
//   p := 1.0 + r*(1.0 + r*(0.5 + r*(0.16666667 + r*(0.04166667 + r*0.00833333)))); // rank 5 approximation

//v.f := p;
//inc(v.i, trunc(n) shl 23);

  Vcvtps2dq     Ymm2  , Ymm2
  Vpslld        Ymm2  , Ymm2    , 23
  Vpaddd        Ymm0  , Ymm0    , Ymm2
  // Inactive tail lanes must not contribute exp(0-max) to the denominator.
  Vandps        Ymm0  , Ymm0    , Ymm4
  Vmaskmovps    Yword[Src]      , Ymm4 , Ymm0
  Vaddps        Ymm7  , Ymm7    , Ymm0

@div:
  Vdpps         Ymm7  , Ymm7, [Rip+ONE], $F1      // sum_exp = sum(exp(src - max_src))
  Vextractf128  Xmm0  , Ymm7    , 1
  // Keep the upper half of the tail mask until normalization completes.
  Vaddss        Xmm0  , Xmm0    , Xmm7

  Vmovss        Xmm3  , Dword [Rip+ONE]
  Vdivss        Xmm3  , Xmm3, Xmm0          // accurate 1/sum_exp

  Mov           Src  , R8
  Vbroadcastss  Ymm3 , Xmm3
  Mov           Rax  , N
  shr           Rax  , 3
  Jz            @Remdiv

@While1:
  Vmulps        Ymm0 , Ymm3, Yword[Src]
  Vmovups       Yword[Src] , Ymm0

  Add           Src  , 32
  Dec           Rax
  Jnz           @While1

@Remdiv:
  Mov           Rax  , N
  and           Rax  , 7
  Jz            @Done
  // we aledy have the mask in ymm4
  Vmaskmovps    Ymm0 , Ymm4, Yword[Src]
  Vmulps        Ymm0,  Ymm0, Ymm3
  Vmaskmovps    Yword[Src] , Ymm4, Ymm0

@Done:
  Vzeroupper
  Vmovdqu              Xmm6     , [Rsp+$00]
  Vmovdqu              Xmm7     , [Rsp+$10]
  Vmovdqu              Xmm8     , [Rsp+$20]
  Vmovdqu              Xmm9     , [Rsp+$30]
  Add                  Rsp      , $10*4                     // restoring stack

end;

function SdotAVX2(const N: Int64; const A, B: System.PSingle): Single; assembler;
asm

  .NOFRAME


   Sub              Rsp      , $10*2                     // making stack space to save one xmm size register
   Vmovdqu          [Rsp+$00], Xmm6
   Vmovdqu          [Rsp+$10], Xmm7

   Vxorps           Ymm0    ,    Ymm0   ,   Ymm0
   Vxorps           Ymm1    ,    Ymm1   ,   Ymm1
   Vxorps           Ymm2    ,    Ymm2   ,   Ymm2
   Vxorps           Ymm3    ,    Ymm3   ,   Ymm3

   Mov              R11     ,    N
   shr              R11     ,    5
   Jz               @Rem1
@while:
   Vmovups          Ymm4    ,    Yword [A]
   Vmovups          Ymm5    ,    Yword [A + 32*1]
   Vmovups          Ymm6    ,    Yword [A + 32*2]
   Vmovups          Ymm7    ,    Yword [A + 32*3]

   Vfmadd231ps      Ymm0    ,    Ymm4   , Yword [B]
   Vfmadd231ps      Ymm1    ,    Ymm5   , Yword [B+32*1]
   Vfmadd231ps      Ymm2    ,    Ymm6   , Yword [B+32*2]
   Vfmadd231ps      Ymm3    ,    Ymm7   , Yword [B+32*3]

   Add              A       ,    32*4
   Add              B       ,    32*4
   Dec              R11
   Jnz              @while

   Vaddps           Ymm0    ,    Ymm0   , Ymm1
   Vaddps           Ymm2    ,    Ymm2   , Ymm3
   Vaddps           Ymm0    ,    Ymm0   , Ymm2


@Rem1:

   Mov              R11     ,    N
   and              R11     ,    31
   shr              R11     ,    3
   Jz               @Rem

@While1:
  Vmovups          Ymm1    ,    Yword [A]
  Vfmadd231ps      Ymm0    ,    Ymm1   , Yword [B]
  Add              A       ,    32
  Add              B       ,    32
  Dec              R11
  Jnz              @While1

@Rem:
   Mov              R11     ,    N
   and              R11     ,    7
   Jz               @Done

   Vmovd            Xmm3    ,    R11d

   Vpbroadcastd     Ymm3    ,    Xmm3
   Vpcmpgtd         Ymm3    ,    Ymm3    , [Rip+YMMD]
   Vmaskmovps       Ymm1    ,    Ymm3    , [A]
   Vmaskmovps       Ymm2    ,    Ymm3    , [B]
   Vfmadd231ps      Ymm0    ,    Ymm1    , Ymm2

@Done:
   Vextractf128     Xmm1    ,    Ymm0   ,   $1
   Vzeroupper
   Vaddps           Xmm0    ,    Xmm0   ,   Xmm1
   //dpps             xmm0    ,    oword [rip+ONE] , $F1
   Haddps           Xmm0    ,    Xmm0
   Haddps           Xmm0    ,    Xmm0

   Vmovdqu          Xmm6    , [Rsp+$00]
   Vmovdqu          Xmm7    , [Rsp+$10]
   Add              Rsp     , $10*2                     // restoring stack

end;

procedure SaxpyAVX2(const N: Int64; const A: Single; const X, Y: System.PSingle); assembler;
asm
  //push         r11
  //push         N

  .NOFRAME


//  movss         xmm2   , a
  Vbroadcastss Ymm1   , A
  Mov          R11    , N
  shr          R11    , (SIMD_SHFT + 2)    // div by 16 (4*4) = turns * SIMD_REGS
  Jz           @Rem1

@while:
  Vmovups      Ymm0   , Yword [Y]
  Vmovups      Ymm2   , Yword [Y+SIMD_OFF]
  Vmovups      Ymm3   , Yword [Y+SIMD_OFF*2]
  Vmovups      Ymm4   , Yword [Y+SIMD_OFF*3]

  Vfmadd231ps  Ymm0   , Ymm1       , Yword [X]                 //xmm0
  Vfmadd231ps  Ymm2   , Ymm1       , Yword [X+SIMD_OFF]  //xmm2
  Vfmadd231ps  Ymm3   , Ymm1       , Yword [X+SIMD_OFF*2]//xmm8
  Vfmadd231ps  Ymm4   , Ymm1       , Yword [X+SIMD_OFF*3]//xmm3

  Vmovups      Yword [Y]             , Ymm0
  Vmovups      Yword [Y+SIMD_OFF]    , Ymm2
  Vmovups      Yword [Y+SIMD_OFF*2]  , Ymm3
  Vmovups      Yword [Y+SIMD_OFF*3]  , Ymm4

  Add          X      , 4 * SIMD_OFF   // turns * offset
  Add          Y      , 4 * SIMD_OFF
  Dec          R11
  Jnz          @while

@Rem1:
  Mov          R11    , N
  and          R11    , (4*SIMD_REGS-1)       // mod 32  ( turns * SIMD_REGS)
  shr          R11    , SIMD_SHFT             // div SIMD_REGS
  Jz           @Rem

@While1:
  Vmovups      Ymm0   , Yword[Y]

  Vfmadd231ps  Ymm0   , Ymm1       , [X]
  Vmovups      Yword [Y]    , Ymm0
  Add          X      , SIMD_OFF
  Add          Y      , SIMD_OFF
  Dec          R11
  Jnz          @While1
@Rem:
  Vzeroupper
  Mov          R11    , N
  and          R11    , (SIMD_REGS -1)       // mod SIMD_REGS
  Jz           @Done

@While2:
  Vmovss       Xmm0   , Dword [Y]
  Vfmadd231ss  Xmm0   , Xmm1, [X]
  Vmovss       Dword [Y]    , Xmm0
  Add          X      , 4
  Add          Y      , 4
  Dec          R11
  Jnz          @While2

@Done:
  //vzeroupper
  //pop          r11
end;

function SSumSqrAVX2(const N: NativeInt; const Src: System.PSingle): Single; assembler;
asm
  Vxorps        Ymm0 , Ymm0, Ymm0
  Vxorps        Ymm1 , Ymm1, Ymm1

  Mov           R8   , N
  shr           R8   , 4
  Jz            @Rem1


@While1:
  Vmovups       Ymm4,  Yword[Src]
  Vmovups       Ymm5,  Yword[Src +32]

  Vfmadd231ps   Ymm0 , Ymm4, Ymm4
  Vfmadd231ps   Ymm1 , Ymm5, Ymm5


  Add           Src  , 32*2
  Dec           R8
  Jnz           @While1

  Vaddps        Ymm0, Ymm0, Ymm1

@Rem1:

  Mov           R8   , N
  and           R8   , 15
  shr           R8   , 3
  Jz            @Rem

@while:
  Vmovups       Ymm1,  Yword[Src]
  Vfmadd231ps   Ymm0 , Ymm1, Ymm1
  Add           Src  , 32
  Dec           R8
  Jnz           @while

@Rem:
  Mov           R8   , N
  and           R8   , 7
  Jz            @Done

  Vmovd         Xmm1 , R8d
  Vpbroadcastd  Ymm1 , Xmm1
  Vpcmpgtd      Ymm1 , Ymm1, YMMD[Rip]
  Vmaskmovps    Ymm2 , Ymm1, Yword[Src]
  //vblendvps     ymm1 , ymm0, yword[src], ymm1
  Vfmadd231ps   Ymm0 , Ymm2, Ymm2

@Done:
  //vbroadcastss  ymm1, [ONE+rip]
  Vdpps         Ymm0 , Ymm0, [Rip+ONE], $F1
  Vextractf128  Xmm1 , Ymm0, 1
  Vzeroupper
  Addss         Xmm0 , Xmm1
  //vextractf128  xmm1 , ymm0, 1
  //vzeroupper
  //addps         xmm0 , xmm1
  //movhlps       xmm1 , xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  //addps         xmm0 , xmm1
  //movshdup      xmm1 , xmm0
  //addss         xmm0 , xmm1

end;

function SSumSqrDiffAVX2(const N: NativeInt; const Src, AMean: System.PSingle): Single; assembler;
asm
  Vbroadcastss  Ymm2 , Dword[AMean]
  Vxorps        Ymm0 , Ymm0, Ymm0
  Vxorps        Ymm1 , Ymm1, Ymm1

  Mov           R9   , N
  shr           R9   , 4
  Jz            @Rem1


@While1:

  Vsubps        Ymm4,  Ymm2, Yword[Src]
  Vsubps        Ymm5,  Ymm2, Yword[Src + 32]

  Vfmadd231ps   Ymm0 , Ymm4, Ymm4
  Vfmadd231ps   Ymm1 , Ymm5, Ymm5


  Add           Src  , 32*2
  Dec           R9
  Jnz           @While1

  Vaddps        Ymm0, Ymm0, Ymm1

@Rem1:

  Mov           R9   , N
  and           R9   , 15
  shr           R9   , 3
  Jz            @Rem

@while:
  Vsubps        Ymm1 , Ymm2, Yword[Src]
  Vfmadd231ps   Ymm0 , Ymm1, Ymm1
  Add           Src  , 32
  Dec           R9
  Jnz           @while

@Rem:
  Mov           R9   , N
  and           R9   , 7
  Jz            @Done

  Vmovd         Xmm1 , R9d
  Vpbroadcastd  Ymm1 , Xmm1
  Vpcmpgtd      Ymm1 , Ymm1, YMMD[Rip]
  Vmaskmovps    Ymm3 , Ymm1, Yword[Src]
  Vsubps        Ymm3 , Ymm2, Ymm3
  Vandps        Ymm3 , Ymm3, Ymm1
  //vblendvps     ymm1 , ymm0, yword[src], ymm1
  Vfmadd231ps   Ymm0 , Ymm3, Ymm3

@Done:
  //vbroadcastss  ymm1, [ONE+rip]
  Vdpps         Ymm0 , Ymm0, [Rip+ONE], $F1
  Vextractf128  Xmm1 , Ymm0, 1
  Vzeroupper
  Addss         Xmm0 , Xmm1
  //vextractf128  xmm1 , ymm0, 1
  //vzeroupper
  //addps         xmm0 , xmm1
  //movhlps       xmm1 , xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  //addps         xmm0 , xmm1
  //movshdup      xmm1 , xmm0
  //addss         xmm0 , xmm1

end;

procedure SFusedMulScaleAVX2(const N: NativeInt; const Dst, A, B: System.PSingle; const Scale: Single); assembler;
asm
  {$IFDEF MSWINDOWS}
  Movss         Xmm4 , Dword[Scale]
  {$ELSE}
  Movss         Xmm4 , Scale
  {$ENDIF}
  Vbroadcastss  Ymm4 , Xmm4

  Mov           Rax  , N
  shr           Rax  , 5
  Jz            @Rem1

@while:
  Vmulps        Ymm0 , Ymm4, Yword[A]
  Vmulps        Ymm1 , Ymm4, Yword[A+32*1]
  Vmulps        Ymm2 , Ymm4, Yword[A+32*2]
  Vmulps        Ymm3 , Ymm4, Yword[A+32*3]

  Vmulps        Ymm0 , Ymm0, Yword[B]
  Vmulps        Ymm1 , Ymm1, Yword[B+32*1]
  Vmulps        Ymm2 , Ymm2, Yword[B+32*2]
  Vmulps        Ymm3 , Ymm3, Yword[B+32*3]

  Vmovups       Yword[Dst]      , Ymm0
  Vmovups       Yword[Dst+32*1] , Ymm1
  Vmovups       Yword[Dst+32*2] , Ymm2
  Vmovups       Yword[Dst+32*3] , Ymm3

  Add           A    , 32*4
  Add           B    , 32*4
  Add           Dst  , 32*4
  Dec           Rax
  Jnz           @while

@Rem1:
  Mov           Rax  , N
  and           Rax  , 31
  shr           Rax  , 3
  Jz            @Rem

@While1:
  Vmulps        Ymm0 , Ymm4, Yword[A]
  Vmulps        Ymm0 , Ymm0, Yword[B]
  Vmovups       Yword[Dst] , Ymm0

  Add           A    , 32
  Add           B    , 32
  Add           Dst  , 32
  Dec           Rax
  Jnz           @While1

@Rem:
  Mov           Rax  , N
  and           Rax  , 7
  Jz            @Done
  Vmovd         Xmm5 , Eax
  Vpbroadcastd  Ymm5 , Xmm5
  Vpcmpgtd      Ymm5 , Ymm5, YMMD[Rip]
  Vmaskmovps    Ymm0 , Ymm5, Yword[A]
  Vmaskmovps    Ymm1 , Ymm5, Yword[B]
  Vmulps        Ymm0,  Ymm0, Ymm4
  Vmulps        Ymm0,  Ymm0, Ymm1
  Vmaskmovps    Yword[Dst] , Ymm5, Ymm0
@Done:
  Vzeroupper
end;

procedure SFusedScaleBiasAVX2(const N: NativeInt; const Dst, Src: System.PSingle; const Scale, Bias: Single); assembler;
asm
  {$IFDEF MSWINDOWS}
  Movss         Xmm5 , Dword[Bias]
  {$ELSE}
  Movss         Xmm5 , Bias
  {$ENDIF}
  Vbroadcastss  Ymm4 , Scale
  Vbroadcastss  Ymm5 , Xmm5

  Mov           Rax  , N
  shr           Rax  , 5
  Jz            @Rem1

@while:

  Vmovups       Ymm0 , Yword [Src]
  Vmovups       Ymm1 , Yword [Src+32*1]
  Vmovups       Ymm2 , Yword [Src+32*2]
  Vmovups       Ymm3 , Yword [Src+32*3]

  Vfmadd132ps   Ymm0 , Ymm5 , Ymm4
  Vfmadd132ps   Ymm1 , Ymm5 , Ymm4
  Vfmadd132ps   Ymm2 , Ymm5 , Ymm4
  Vfmadd132ps   Ymm3 , Ymm5 , Ymm4

  Vmovups       Yword[Dst]      , Ymm0
  Vmovups       Yword[Dst+32*1] , Ymm1
  Vmovups       Yword[Dst+32*2] , Ymm2
  Vmovups       Yword[Dst+32*3] , Ymm3

  Add           Src  , 32*4
  Add           Dst  , 32*4
  Dec           Rax
  Jnz           @while

@Rem1:
  Mov           Rax  , N
  and           Rax  , 31
  shr           Rax  , 3
  Jz            @Rem

@While1:
  Vmovups       Ymm0 , Yword [Src]
  Vfmadd132ps   Ymm0 , Ymm5 , Ymm4
  Vmovups       Yword[Dst]  , Ymm0

  Add           Src  , 32
  Add           Dst  , 32
  Dec           Rax
  Jnz           @While1

@Rem:
  Mov           Rax  , N
  and           Rax  , 7
  Jz            @Done
  Vmovd         Xmm1 , Eax
  Vpbroadcastd  Ymm1 , Xmm1
  Vpcmpgtd      Ymm1 , Ymm1, YMMD[Rip]
  Vmaskmovps    Ymm0 , Ymm1, Yword[Src]
  Vfmadd132ps   Ymm0 , Ymm5 , Ymm4
  Vmaskmovps    Yword[Dst] , Ymm1, Ymm0
@Done:
  Vzeroupper
end;

procedure SScaleAVX2(const N: NativeInt; Dst, Src: System.PSingle; const Scale: Single); assembler;
asm

    Vbroadcastss  Ymm4 , Scale

    Mov           Rax  , N
    shr           Rax  , 5
    Jz            @Rem1

  @while:

    Vmulps        Ymm0 , Ymm4 , Yword [Src]
    Vmulps        Ymm1 , Ymm4 , Yword [Src+32*1]
    Vmulps        Ymm2 , Ymm4 , Yword [Src+32*2]
    Vmulps        Ymm3 , Ymm4 , Yword [Src+32*3]

    Vmovups       Yword[Dst]      , Ymm0
    Vmovups       Yword[Dst+32*1] , Ymm1
    Vmovups       Yword[Dst+32*2] , Ymm2
    Vmovups       Yword[Dst+32*3] , Ymm3

    Add           Src  , 32*4
    Add           Dst  , 32*4
    Dec           Rax
    Jnz           @while

  @Rem1:
    Mov           Rax  , N
    and           Rax  , 31
    shr           Rax  , 3
    Jz            @Rem

  @While1:

    Vmulps        Ymm0 , Ymm4 , Yword [Src]
    Vmovups       Yword[Dst]  , Ymm0

    Add           Src  , 32
    Add           Dst  , 32
    Dec           Rax
    Jnz           @While1

  @Rem:
    Mov           Rax  , N
    and           Rax  , 7
    Jz            @Done
    Vmovd         Xmm1 , Eax
    Vpbroadcastd  Ymm1 , Xmm1
    Vpcmpgtd      Ymm1 , Ymm1, YMMD[Rip]
    Vmaskmovps    Ymm0 , Ymm1, Yword[Src]
    Vmulps        Ymm0 , Ymm4, Ymm0
    Vmaskmovps    Yword[Dst] , Ymm1, Ymm0
  @Done:
    Vzeroupper
end;

(*
  performs :
  const T : array [0..7] of single = (0, 0, 0, 0, 0, 0, 0, 0);
  begin

    for j:=0 to K-1 do begin
      for i:=0 to 7 do begin
        t[i] += A[j]*B[j + i*ldb]
      end;
    end
    for i :=0 to 7 do
      C[i] += t[i]
  end
*)
procedure SDotX8AVX2(const A, B, C: System.PSingle; K, Ldb: NativeInt; const Alpha: Single); assembler;
asm
  Sub           Rsp  , $10*7
  Vmovdqu       [Rsp + $00] , Xmm6
  Vmovdqu       [Rsp + $01] , Xmm7
  Vmovdqu       [Rsp + $02] , Xmm8
  Vmovdqu       [Rsp + $03] , Xmm9
  Vmovdqu       [Rsp + $04] , Xmm10
  Vmovdqu       [Rsp + $05] , Xmm11
  Vmovdqu       [Rsp + $06] , Xmm12


  Vxorps        Ymm1 , Ymm1 , Ymm1
  Vxorps        Ymm2 , Ymm2 , Ymm2
  Vxorps        Ymm3 , Ymm3 , Ymm3
  Vxorps        Ymm4 , Ymm4 , Ymm4
  Vxorps        Ymm5 , Ymm5 , Ymm5
  Vxorps        Ymm6 , Ymm6 , Ymm6
  Vxorps        Ymm7 , Ymm7 , Ymm7
  Vxorps        Ymm8 , Ymm8 , Ymm8

  xor           R10  , R10
  Mov           R11  , K
  shr           R11  , 3  // div 8
  Jz            @Rem
@LoopK:
  Vmovups       Ymm0 , Yword [A + R10*4]

  Mov           Rax  , R10

  Vfmadd231ps   Ymm1 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm2 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm3 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm4 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm5 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm6 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm7 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm8 , Ymm0 ,Yword[B + Rax*4]

  Add           R10  , 8
  Dec           R11
  Jnz           @LoopK

@Rem:
  Mov           Rax  , K
  and           Rax  , 7
  Jz            @Final
  Vmovd         Xmm5 , Eax
  Vpbroadcastd  Ymm5 , Xmm5
  Vpcmpgtd      Ymm5 , Ymm5 , YMMD[Rip]

  Vmaskmovps    Ymm0 , Ymm5 , Yword [A + R10*4]
  Mov           Rax  , R10

  Vmaskmovps    Ymm1 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm2 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm3 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm4 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm5 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm7 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm8 , Ymm5 , Yword[B + Rax*4]

  Vfmadd231ps   Ymm1 , Ymm0 ,Ymm1
  Vfmadd231ps   Ymm2 , Ymm0 ,Ymm2
  Vfmadd231ps   Ymm3 , Ymm0 ,Ymm3
  Vfmadd231ps   Ymm4 , Ymm0 ,Ymm4
  Vfmadd231ps   Ymm5 , Ymm0 ,Ymm5
  Vfmadd231ps   Ymm6 , Ymm0 ,Ymm6
  Vfmadd231ps   Ymm7 , Ymm0 ,Ymm7
  Vfmadd231ps   Ymm8 , Ymm0 ,Ymm8


@Final:
  Vextractf128  Xmm9  , Ymm1 , $1
  Vextractf128  Xmm10 , Ymm2 , $1
  Vextractf128  Xmm11 , Ymm3 , $1
  Vextractf128  Xmm12 , Ymm4 , $1
  Addps         Xmm1  , Xmm9
  Addps         Xmm2  , Xmm10
  Addps         Xmm3  , Xmm11
  Addps         Xmm4  , Xmm12

  Vextractf128  Xmm9  , Ymm5  , $1
  Vextractf128  Xmm10 , Ymm6  , $1
  Vextractf128  Xmm11 , Ymm7  , $1
  Vextractf128  Xmm12 , Ymm8  , $1
  Vzeroupper
  Addps         Xmm5  , Xmm9
  Addps         Xmm6  , Xmm10
  Addps         Xmm7  , Xmm11
  Addps         Xmm8  , Xmm12


  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2
  Haddps        Xmm3 , Xmm3
  Haddps        Xmm4 , Xmm4
  Haddps        Xmm5 , Xmm5
  Haddps        Xmm6 , Xmm6
  Haddps        Xmm7 , Xmm7
  Haddps        Xmm8 , Xmm8

  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2
  Haddps        Xmm3 , Xmm3
  Haddps        Xmm4 , Xmm4
  Haddps        Xmm5 , Xmm5
  Haddps        Xmm6 , Xmm6
  Haddps        Xmm7 , Xmm7
  Haddps        Xmm8 , Xmm8

  Insertps      Xmm0 , Xmm5 , $00
  Insertps      Xmm0 , Xmm6 , $10
  Insertps      Xmm0 , Xmm7 , $20
  Insertps      Xmm0 , Xmm8 , $30
  Vinsertf128   Ymm0 , Ymm0 , Xmm0, $1

  Insertps      Xmm0 , Xmm1 , $00
  Insertps      Xmm0 , Xmm2 , $10
  Insertps      Xmm0 , Xmm3 , $20
  Insertps      Xmm0 , Xmm4 , $30

  Movss         Xmm1 , Alpha
  Vbroadcastss  Ymm1 , Xmm1
  //vmulps        ymm0 , ymm0 , ymm1
  //vaddps        ymm0 , ymm0 , yword[C]
  Vfmadd213ps   Ymm0 , Ymm1 , Yword[C]
  Vmovups       Yword[C], Ymm0

  Vzeroupper
  Vmovdqu       Xmm6 , [Rsp + $00]
  Vmovdqu       Xmm7 , [Rsp + $01]
  Vmovdqu       Xmm8 , [Rsp + $02]
  Vmovdqu       Xmm9 , [Rsp + $03]
  Vmovdqu       Xmm10, [Rsp + $04]
  Vmovdqu       Xmm11, [Rsp + $05]
  Vmovdqu       Xmm12, [Rsp + $06]
  Add           Rsp  , $10*7
end;

procedure SDotX4AVX2(const A, B, C: System.PSingle; K, Ldb: NativeInt; const Alpha: Single); assembler;
asm
  Sub           Rsp  , $10*4
  Vmovdqu       [Rsp + $00] , Xmm6
  Vmovdqu       [Rsp + $10] , Xmm7
  Vmovdqu       [Rsp + $20] , Xmm8
  Vmovdqu       [Rsp + $30] , Xmm9


  Vxorps        Ymm1 , Ymm1 , Ymm1
  Vxorps        Ymm2 , Ymm2 , Ymm2
  Vxorps        Ymm3 , Ymm3 , Ymm3
  Vxorps        Ymm4 , Ymm4 , Ymm4

  xor           R10  , R10
  Mov           R11  , K
  shr           R11  , 3  // div 8
  Jz            @Rem
@LoopK:
  Vmovups       Ymm0 , Yword [A + R10*4]

  Mov           Rax  , R10

  Vfmadd231ps   Ymm1 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm2 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm3 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm4 , Ymm0 ,Yword[B + Rax*4]

  Add           R10  , 8
  Dec           R11
  Jnz           @LoopK

@Rem:
  Mov           Rax  , K
  and           Rax  , 7
  Jz            @Final
  Vmovd         Xmm5 , Eax
  Vpbroadcastd  Ymm5 , Xmm5
  Vpcmpgtd      Ymm5 , Ymm5 , YMMD[Rip]

  Vmaskmovps    Ymm0 , Ymm5 , Yword [A + R10*4]
  Mov           Rax  , R10

  Vmaskmovps    Ymm6 , Ymm5 , Yword[B + Rax*4]
  Vfmadd231ps   Ymm1 , Ymm0 , Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 , Yword[B + Rax*4]
  Vfmadd231ps   Ymm2 , Ymm0 , Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 , Yword[B + Rax*4]
  Vfmadd231ps   Ymm3 , Ymm0 , Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 , Yword[B + Rax*4]
  Vfmadd231ps   Ymm4 , Ymm0 , Ymm6

@Final:
  Vextractf128  Xmm6 , Ymm1 , $1
  Vextractf128  Xmm7 , Ymm2 , $1
  Vextractf128  Xmm8 , Ymm3 , $1
  Vextractf128  Xmm9 , Ymm4 , $1
  Vzeroupper
  Addps         Xmm1  , Xmm6
  Addps         Xmm2  , Xmm7
  Addps         Xmm3  , Xmm8
  Addps         Xmm4  , Xmm9


  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2
  Haddps        Xmm3 , Xmm3
  Haddps        Xmm4 , Xmm4

  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2
  Haddps        Xmm3 , Xmm3
  Haddps        Xmm4 , Xmm4

  Insertps      Xmm0 , Xmm1 , $00
  Insertps      Xmm0 , Xmm2 , $10
  Insertps      Xmm0 , Xmm3 , $20
  Insertps      Xmm0 , Xmm4 , $30

  Movss         Xmm1 , Alpha
  Vbroadcastss  Xmm1 , Xmm1
  //vmulps        ymm0 , ymm0 , ymm1
  //vaddps        ymm0 , ymm0 , yword[C]
  Vfmadd213ps   Xmm0 , Xmm1 , Oword[C]
  Vmovups       Oword[C], Xmm0


  Vmovdqu       Xmm6 , [Rsp + $00]
  Vmovdqu       Xmm7 , [Rsp + $10]
  Vmovdqu       Xmm8 , [Rsp + $20]
  Vmovdqu       Xmm9 , [Rsp + $30]
  Add           Rsp  , $10*4
end;

procedure SDot4x4AVX2(const A, B, C: System.PSingle; K, Lda, Ldb, Ldc: NativeInt; const Alpha: Single); assembler;
var
  C4x4: array[0..4 * 4 * 8 - 1] of Single;
asm
  Push          R12
  Sub           Rsp  , $10*4
  Vmovdqu       [Rsp + $00] , Xmm6
  Vmovdqu       [Rsp + $01] , Xmm7
  Vmovdqu       [Rsp + $02] , Xmm8
  Vmovdqu       [Rsp + $03] , Xmm9

  Vxorps        Ymm1 , Ymm1 , Ymm1
  Vxorps        Ymm2 , Ymm2 , Ymm2
  Vxorps        Ymm3 , Ymm3 , Ymm3
  Vxorps        Ymm4 , Ymm4 , Ymm4
  xor           Rax              , Rax
@ResetC:
  Vmovups       Yword [C4x4 + Rax*4]         , Ymm1
  Vmovups       Yword [C4x4 + Rax*4 + $20]   , Ymm2
  Vmovups       Yword [C4x4 + Rax*4 + $40]   , Ymm3
  Vmovups       Yword [C4x4 + Rax*4 + $60]   , Ymm4
  Add           Rax                    , $8*4
  Cmp           Rax                    , 4*4*8
  Jl            @ResetC


  xor           R12  , R12
  Mov           R11  , K
  shr           R11  , 3  // div 8
  Jz            @Rem

@LoopK:
  Mov           R10  , R12
  Vmovups       Ymm0 , Yword [A + R10*4]
  Mov           Rax  , R12
  Vmovups       Ymm1 , Yword [C4x4       ]
  Vmovups       Ymm2 , Yword [C4x4 + $20 ]
  Vmovups       Ymm3 , Yword [C4x4 + $40 ]
  Vmovups       Ymm4 , Yword [C4x4 + $60 ]

  Vfmadd231ps   Ymm1 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm2 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm3 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm4 , Ymm0 ,Yword[B + Rax*4]
  Vmovups       Yword [C4x4       ]       ,Ymm1
  Vmovups       Yword [C4x4 + $20 ]       ,Ymm2
  Vmovups       Yword [C4x4 + $40 ]       ,Ymm3
  Vmovups       Yword [C4x4 + $60 ]       ,Ymm4
  Add           R10  , Lda

  Vmovups       Ymm0 , Yword [A + R10*4]
  Mov           Rax  , R12
  Vmovups       Ymm1 , Yword [C4x4 + $80]
  Vmovups       Ymm2 , Yword [C4x4 + $A0]
  Vmovups       Ymm3 , Yword [C4x4 + $C0]
  Vmovups       Ymm4 , Yword [C4x4 + $E0]

  Vfmadd231ps   Ymm1 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm2 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm3 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm4 , Ymm0 ,Yword[B + Rax*4]
  Vmovups       Yword [C4x4 + $80]       ,Ymm1
  Vmovups       Yword [C4x4 + $A0]       ,Ymm2
  Vmovups       Yword [C4x4 + $C0]       ,Ymm3
  Vmovups       Yword [C4x4 + $E0]       ,Ymm4
  Add           R10  , Lda

  Vmovups       Ymm0 , Yword [A + R10*4]
  Mov           Rax  , R12
  Vmovups       Ymm1 , Yword [C4x4 + $100]
  Vmovups       Ymm2 , Yword [C4x4 + $120]
  Vmovups       Ymm3 , Yword [C4x4 + $140]
  Vmovups       Ymm4 , Yword [C4x4 + $160]

  Vfmadd231ps   Ymm1 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm2 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm3 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm4 , Ymm0 ,Yword[B + Rax*4]
  Vmovups       Yword [C4x4 + $100]       ,Ymm1
  Vmovups       Yword [C4x4 + $120]       ,Ymm2
  Vmovups       Yword [C4x4 + $140]       ,Ymm3
  Vmovups       Yword [C4x4 + $160]       ,Ymm4
  Add           R10  , Lda

  Vmovups       Ymm0 , Yword [A + R10*4]
  Mov           Rax  , R12
  Vmovups       Ymm1 , Yword [C4x4 + $180]
  Vmovups       Ymm2 , Yword [C4x4 + $1A0]
  Vmovups       Ymm3 , Yword [C4x4 + $1C0]
  Vmovups       Ymm4 , Yword [C4x4 + $1E0]

  Vfmadd231ps   Ymm1 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm2 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm3 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm4 , Ymm0 ,Yword[B + Rax*4]
  Vmovups       Yword [C4x4 + $180]       ,Ymm1
  Vmovups       Yword [C4x4 + $1A0]       ,Ymm2
  Vmovups       Yword [C4x4 + $1C0]       ,Ymm3
  Vmovups       Yword [C4x4 + $1E0]       ,Ymm4
  Add           R12  , 8

  Dec           R11
  Jnz           @LoopK


@Rem:
  Mov           Rax  , K
  and           Rax  , 7
  Jz            @Final
  Vmovd         Xmm5 , Eax
  Vpbroadcastd  Ymm5 , Xmm5
  Vpcmpgtd      Ymm5 , Ymm5 , YMMD[Rip]

  Mov           R10  , R12
  Vmaskmovps    Ymm0 , Ymm5 , Yword [A + R10*4]

  Vmovups       Ymm1 , Yword [C4x4       ]
  Vmovups       Ymm2 , Yword [C4x4 + $20 ]
  Vmovups       Ymm3 , Yword [C4x4 + $40 ]
  Vmovups       Ymm4 , Yword [C4x4 + $60 ]

  Mov           Rax  , R12
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm1 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm2 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm3 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm4 , Ymm0 ,Ymm6
  Vmovups       Yword [C4x4       ]       ,Ymm1
  Vmovups       Yword [C4x4 + $20 ]       ,Ymm2
  Vmovups       Yword [C4x4 + $40 ]       ,Ymm3
  Vmovups       Yword [C4x4 + $60 ]       ,Ymm4
  Add           R10  , Lda

  Vmovups       Ymm0 , Yword [A + R10*4]
  Mov           Rax  , R12
  Vmovups       Ymm1 , Yword [C4x4 + $80]
  Vmovups       Ymm2 , Yword [C4x4 + $A0]
  Vmovups       Ymm3 , Yword [C4x4 + $C0]
  Vmovups       Ymm4 , Yword [C4x4 + $E0]

  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm1 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm2 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm3 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm4 , Ymm0 ,Ymm6
  Vmovups       Yword [C4x4 + $80]       ,Ymm1
  Vmovups       Yword [C4x4 + $A0]       ,Ymm2
  Vmovups       Yword [C4x4 + $C0]       ,Ymm3
  Vmovups       Yword [C4x4 + $E0]       ,Ymm4
  Add           R10  , Lda

  Vmovups       Ymm0 , Yword [A + R10*4]
  Mov           Rax  , R12
  Vmovups       Ymm1 , Yword [C4x4 + $100]
  Vmovups       Ymm2 , Yword [C4x4 + $120]
  Vmovups       Ymm3 , Yword [C4x4 + $140]
  Vmovups       Ymm4 , Yword [C4x4 + $160]

  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm1 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm2 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm3 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm4 , Ymm0 ,Ymm6
  Vmovups       Yword [C4x4 + $100]       ,Ymm1
  Vmovups       Yword [C4x4 + $120]       ,Ymm2
  Vmovups       Yword [C4x4 + $140]       ,Ymm3
  Vmovups       Yword [C4x4 + $160]       ,Ymm4
  Add           R10  , Lda

  Vmovups       Ymm0 , Yword [A + R10*4]
  Mov           Rax  , R12
  Vmovups       Ymm1 , Yword [C4x4 + $180]
  Vmovups       Ymm2 , Yword [C4x4 + $1A0]
  Vmovups       Ymm3 , Yword [C4x4 + $1C0]
  Vmovups       Ymm4 , Yword [C4x4 + $1E0]

  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm1 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm2 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm3 , Ymm0 ,Ymm6
  Add           Rax  , Ldb
  Vmaskmovps    Ymm6 , Ymm5 ,Yword[B + Rax*4]
  Vfmadd231ps   Ymm4 , Ymm0 ,Ymm6
  Vmovups       Yword [C4x4 + $180]       ,Ymm1
  Vmovups       Yword [C4x4 + $1A0]       ,Ymm2
  Vmovups       Yword [C4x4 + $1C0]       ,Ymm3
  Vmovups       Yword [C4x4 + $1E0]       ,Ymm4

  xor           Rax  , Rax
  xor           R10  , R10

@Final:

  Vmovups       Ymm1 , Yword [C4x4 + Rax]
  Vmovups       Ymm2 , Yword [C4x4 + Rax + $20]
  Vmovups       Ymm3 , Yword [C4x4 + Rax + $40]
  Vmovups       Ymm4 , Yword [C4x4 + Rax + $60]

  Vextractf128  Xmm6 , Ymm1 , $1
  Vextractf128  Xmm7 , Ymm2 , $1
  Vextractf128  Xmm8 , Ymm3 , $1
  Vextractf128  Xmm9 , Ymm4 , $1
  Vzeroupper
  Addps         Xmm1  , Xmm6
  Addps         Xmm2  , Xmm7
  Addps         Xmm3  , Xmm8
  Addps         Xmm4  , Xmm9

  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2
  Haddps        Xmm3 , Xmm3
  Haddps        Xmm4 , Xmm4

  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2
  Haddps        Xmm3 , Xmm3
  Haddps        Xmm4 , Xmm4

  Insertps      Xmm0 , Xmm1 , $00
  Insertps      Xmm0 , Xmm2 , $10
  Insertps      Xmm0 , Xmm3 , $20
  Insertps      Xmm0 , Xmm4 , $30

  Movss         Xmm1 , Alpha
  Vbroadcastss  Xmm1 , Xmm1
  //vmulps        ymm0 , ymm0 , ymm1
  //vaddps        ymm0 , ymm0 , yword[C]
  Vfmadd213ps   Xmm0 , Xmm1 , Oword[C + R10*4]
  Vmovups       Oword[C + R10*4], Xmm0
  Add           R10 , Ldc
  Add           Rax , $80
  Cmp           Rax , $200
  Jl            @Final
@Done:
  Vmovdqu       Xmm6 , [Rsp + $00]
  Vmovdqu       Xmm7 , [Rsp + $01]
  Vmovdqu       Xmm8 , [Rsp + $02]
  Vmovdqu       Xmm9 , [Rsp + $03]
  Add           Rsp  , $10*4
  Pop           R12
end;

procedure SDotX2AVX2(const A, B, C: System.PSingle; K, Ldb: NativeInt; const Alpha: Single); assembler;
asm


  Vxorps        Ymm1 , Ymm1 , Ymm1
  Vxorps        Ymm2 , Ymm2 , Ymm2

  xor           R10  , R10
  Mov           R11  , K
  shr           R11  , 3  // div 8
  Jz            @Rem
@LoopK:
  Vmovups       Ymm0 , Yword [A + R10*4]

  Mov           Rax  , R10

  Vfmadd231ps   Ymm1 , Ymm0 ,Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vfmadd231ps   Ymm2 , Ymm0 ,Yword[B + Rax*4]

  Add           R10  , 8
  Dec           R11
  Jnz           @LoopK

@Rem:
  Mov           Rax  , K
  and           Rax  , 7
  Jz            @Final
  Vmovd         Xmm5 , Eax
  Vpbroadcastd  Ymm5 , Xmm5
  Vpcmpgtd      Ymm5 , Ymm5 , YMMD[Rip]

  Vmaskmovps    Ymm0 , Ymm5 , Yword [A + R10*4]
  Mov           Rax  , R10

  Vmaskmovps    Ymm1 , Ymm5 , Yword[B + Rax*4]
  Add           Rax  , Ldb
  Vmaskmovps    Ymm2 , Ymm5 , Yword[B + Rax*4]

  Vfmadd231ps   Ymm1 , Ymm0 ,Ymm1
  Vfmadd231ps   Ymm2 , Ymm0 ,Ymm2


@Final:
  Vextractf128  Xmm3 , Ymm1 , $1
  Vextractf128  Xmm4 , Ymm2 , $1

  Vzeroupper
  Addps         Xmm1  , Xmm3
  Addps         Xmm2  , Xmm4


  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2

  Haddps        Xmm1 , Xmm1
  Haddps        Xmm2 , Xmm2

  //insertps      xmm0 , xmm1 , $00
  //insertps      xmm0 , xmm2 , $10
  //insertps      xmm0 , xmm3 , $20
  //insertps      xmm0 , xmm4 , $30

  Movss         Xmm0 , Alpha
  //vbroadcastss  xmm1 , xmm1
  //vmulps        ymm0 , ymm0 , ymm1
  //vaddps        ymm0 , ymm0 , yword[C]
  Vfmadd213ss   Xmm1 , Xmm0 , Dword[C]
  Movss         Dword[C], Xmm1
  Vfmadd213ss   Xmm2 , Xmm0 , Dword[C+4]
  Movss         Dword[C+4], Xmm2

end;

procedure SingleToBF16X8AVX2(const Src: System.PSingle; const Dst: PBF16); assembler;
asm
  Vmovups       Ymm0        , Yword [Src]
  Vpsrld        Ymm0        , Ymm0 , 16
  Vextractf128  Xmm1        , Ymm0 , 1
  Vpackusdw     Xmm0        , Xmm0 , Xmm1
  Vmovups       Oword [Dst] , Xmm0
end;

procedure BF16ToSingleX8AVX2(const Src: PBF16; const Dst: System.PSingle); assembler;
asm
  Vpmovzxwd     Ymm0        , Oword [Src]
  Vpslld        Ymm0        , Ymm0 , 16
  Vmovups       Yword [Dst] , Ymm0
end;

// test
procedure SDot4x4(const A, B, C: System.PSingle; K, Lda, Ldb, Ldc: NativeInt; const Alpha: Single);
var
  I, J: Int64;
  CC: System.PSingle;
begin
  for I := 0 to 3 do
  begin
    CC := C + I * Ldc;
    for J := 0 to 3 do
    begin
      CC[J] := CC[J] + Alpha * SdotAVX2(K, A + I * Lda, B + J * Ldb);
    end
  end;
end;

{$ENDIF}

// function fast_Exp( x:single):single;
// var
//  t1, t2:single;
//  t0 : longint;
// begin
//  if x>MAX_EXP then exit(1e38);
//  if x<MIN_EXP then exit(0.0);
//  x  := x*L2E;
//  t0 := floor(x);
//  x  := x - t0;
//  x  := x*(C0*x + C1)  +  C2;
//  t0 :=(t0 shl 23) + PLongint(@x)^;
//  result := PSingle(@t0)^;
// end;

procedure QNNExp(const N: NativeInt; const Src: System.PSingle);
var
  I: NativeInt;
begin
  for I := 0 to N - 1 do
    Src[I] := Exp(Src[I])
end;
