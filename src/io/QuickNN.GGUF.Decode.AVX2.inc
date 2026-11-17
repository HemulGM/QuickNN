// Volatile registers only; Win64 nonvolatile XMM registers are untouched.
{$IF defined(CPUX64) and defined(USE_AVX2)}
const
  DECODE_MASK_15: array[0..7] of Integer = (15,15,15,15,15,15,15,15);
  IQ4_BYTES: array[0..15] of ShortInt = (-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);

procedure DecodeLow4(const Src: PByte; const Dst: System.PSingle; const Params: PGGUFDecodeParams); assembler;
asm
  .NOFRAME
  vbroadcastss ymm4, dword [r8]
  vbroadcastss ymm5, dword [r8+4]
  vmovd xmm2, dword [r8+8]
  vmovdqu ymm3, [rip+DECODE_MASK_15]
  mov r9d, 4
@Loop:
  vpmovzxbd ymm0, qword [rcx]
  vpsrld ymm0, ymm0, xmm2
  vpand ymm0, ymm0, ymm3
  vcvtdq2ps ymm0, ymm0
  vmulps ymm0, ymm0, ymm4
  vaddps ymm0, ymm0, ymm5
  vmovups [rdx], ymm0
  add rcx, 8
  add rdx, 32
  dec r9d
  jnz @Loop
  vzeroupper
end;

procedure DecodeBits(const Low, High: PByte; const Dst: System.PSingle; const Params: PGGUFDecodeParams); assembler;
asm
  .NOFRAME
  vbroadcastss ymm4, dword [r9]
  vbroadcastss ymm5, dword [r9+4]
  mov r10d, dword [r9+32]
  shr r10d, 3
@Loop:
  vpmovzxbd ymm0, qword [rcx]
  vmovd xmm2, dword [r9+8]
  vpsrld ymm0, ymm0, xmm2
  vpbroadcastd ymm3, dword [r9+16]
  vpand ymm0, ymm0, ymm3
  test rdx, rdx
  jz @LowOnly
  vpmovzxbd ymm1, qword [rdx]
  vmovd xmm2, dword [r9+12]
  vpsrld ymm1, ymm1, xmm2
  vpbroadcastd ymm3, dword [r9+20]
  vpand ymm1, ymm1, ymm3
  vmovd xmm2, dword [r9+24]
  vpslld ymm1, ymm1, xmm2
  vpor ymm0, ymm0, ymm1
  add rdx, 8
@LowOnly:
  vpbroadcastd ymm3, dword [r9+28]
  vpsubd ymm0, ymm0, ymm3
  vcvtdq2ps ymm0, ymm0
  vmulps ymm0, ymm0, ymm4
  vaddps ymm0, ymm0, ymm5
  vmovups [r8], ymm0
  add rcx, 8
  add r8, 32
  dec r10d
  jnz @Loop
  vzeroupper
end;

procedure DecodeSigned32(const Src: PByte; const Dst: System.PSingle; const Scale: Single); assembler;
asm
  .NOFRAME
  vbroadcastss ymm4, xmm2
  mov r9d, 4
@Loop:
  vpmovsxbd ymm0, qword [rcx]
  vcvtdq2ps ymm0, ymm0
  vmulps ymm0, ymm0, ymm4
  vmovups [rdx], ymm0
  add rcx, 8
  add rdx, 32
  dec r9d
  jnz @Loop
  vzeroupper
end;

procedure DecodeIQ4(const Src: PByte; const Dst: System.PSingle; const Scale: Single); assembler;
asm
  .NOFRAME
  vbroadcastss ymm4, xmm2
  vmovdqu xmm5, [rcx]
  vmovdqu xmm3, [rip+IQ4_BYTES]
  vpcmpeqb xmm1, xmm1, xmm1
  vpsrlw xmm1, xmm1, 12
  vpackuswb xmm1, xmm1, xmm1
  vpand xmm0, xmm5, xmm1
  vpsrlw xmm5, xmm5, 4
  vpand xmm5, xmm5, xmm1
  vpshufb xmm0, xmm3, xmm0
  vpshufb xmm5, xmm3, xmm5
  vpmovsxbd ymm1, xmm0
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmovups [rdx], ymm1
  vpsrldq xmm0, xmm0, 8
  vpmovsxbd ymm1, xmm0
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmovups [rdx+32], ymm1
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmovups [rdx+64], ymm1
  vpsrldq xmm5, xmm5, 8
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmovups [rdx+96], ymm1
  vzeroupper
end;
{$ENDIF}
