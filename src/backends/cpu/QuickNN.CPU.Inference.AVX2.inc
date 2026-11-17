{$IF defined(CPUX64) and defined(USE_AVX2)}
const
  INT4_SHIFTS: array[0..7] of Integer = (0,4,8,12,16,20,24,28);
  INT4_MASK: array[0..7] of Integer = (15,15,15,15,15,15,15,15);
  INT4_CENTER: array[0..7] of Integer = (8,8,8,8,8,8,8,8);

function VectorDot(const Count: Integer; const A, B: System.PSingle): Single; assembler;
asm
  .NOFRAME
  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3
  mov r10d, ecx
  shr r10d, 5
  jz @Tail8
@Loop:
  vmovups ymm4, [rdx]
  vmulps ymm4, ymm4, [r8]
  vaddps ymm0, ymm0, ymm4
  vmovups ymm4, [rdx+32]
  vmulps ymm4, ymm4, [r8+32]
  vaddps ymm1, ymm1, ymm4
  vmovups ymm4, [rdx+64]
  vmulps ymm4, ymm4, [r8+64]
  vaddps ymm2, ymm2, ymm4
  vmovups ymm4, [rdx+96]
  vmulps ymm4, ymm4, [r8+96]
  vaddps ymm3, ymm3, ymm4
  add rdx, 128
  add r8, 128
  dec r10d
  jnz @Loop
@Tail8:
  vaddps ymm0, ymm0, ymm1
  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm2
  mov r10d, ecx
  and r10d, 31
  shr r10d, 3
  jz @Reduce
@Loop8:
  vmovups ymm4, [rdx]
  vmulps ymm4, ymm4, [r8]
  vaddps ymm0, ymm0, ymm4
  add rdx, 32
  add r8, 32
  dec r10d
  jnz @Loop8
@Reduce:
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  and ecx, 7
  jz @Done
@Scalar:
  vmovss xmm1, dword [rdx]
  vmulss xmm1, xmm1, dword [r8]
  vaddss xmm0, xmm0, xmm1
  add rdx, 4
  add r8, 4
  dec ecx
  jnz @Scalar
@Done:
  vzeroupper
end;

procedure ExpandInt4(const Src: PByte; const Dst: System.PSingle); assembler;
asm
  .NOFRAME
  vbroadcastss ymm4, dword [rcx]
  vmovdqu ymm3, [rip+INT4_SHIFTS]
  mov r9d, 4
  add rcx, 4
@Loop:
  vpbroadcastd ymm0, dword [rcx]
  vpsrlvd ymm0, ymm0, ymm3
  vpand ymm0, ymm0, [rip+INT4_MASK]
  vpsubd ymm0, ymm0, [rip+INT4_CENTER]
  vcvtdq2ps ymm0, ymm0
  vmulps ymm0, ymm0, ymm4
  vmovups [rdx], ymm0
  add rcx, 4
  add rdx, 32
  dec r9d
  jnz @Loop
  vzeroupper
end;

procedure MemoryAccumulate(const Count: Integer; const State, Memory: System.PSingle; const Factors: System.PSingle); assembler;
asm
  .NOFRAME
  vbroadcastss ymm4, dword [r9]
  vbroadcastss ymm5, dword [r9+4]
  mov r10d, ecx
  shr r10d, 3
  jz @Tail
@Loop:
  vmovups ymm0, [rdx]
  vmulps ymm0, ymm0, ymm4
  vmulps ymm0, ymm0, ymm5
  vaddps ymm0, ymm0, [r8]
  vmovups [r8], ymm0
  add rdx, 32
  add r8, 32
  dec r10d
  jnz @Loop
@Tail:
  and ecx, 7
  jz @Done
@Scalar:
  vmovss xmm0, dword [rdx]
  vmulss xmm0, xmm0, xmm4
  vmulss xmm0, xmm0, xmm5
  vaddss xmm0, xmm0, dword [r8]
  vmovss dword [r8], xmm0
  add rdx, 4
  add r8, 4
  dec ecx
  jnz @Scalar
@Done:
  vzeroupper
end;

procedure StateUpdate(const Count: Integer; const State, Delta, Output: System.PSingle; const Factors: System.PSingle); assembler;
asm
  .NOFRAME
  mov r10, Factors
  vbroadcastss ymm3, dword [r10]
  vbroadcastss ymm4, dword [r10+4]
  vbroadcastss ymm5, dword [r10+8]
  mov r10d, ecx
  shr r10d, 3
  jz @Tail
@Loop:
  vmovups ymm0, [rdx]
  vmulps ymm0, ymm0, ymm3
  vmovups ymm1, [r8]
  vmulps ymm1, ymm1, ymm4
  vaddps ymm0, ymm0, ymm1
  vmovups [rdx], ymm0
  vmulps ymm0, ymm0, ymm5
  vaddps ymm0, ymm0, [r9]
  vmovups [r9], ymm0
  add rdx, 32
  add r8, 32
  add r9, 32
  dec r10d
  jnz @Loop
@Tail:
  and ecx, 7
  jz @Done
@Scalar:
  vmovss xmm0, dword [rdx]
  vmulss xmm0, xmm0, xmm3
  vmovss xmm1, dword [r8]
  vmulss xmm1, xmm1, xmm4
  vaddss xmm0, xmm0, xmm1
  vmovss dword [rdx], xmm0
  vmulss xmm0, xmm0, xmm5
  vaddss xmm0, xmm0, dword [r9]
  vmovss dword [r9], xmm0
  add rdx, 4
  add r8, 4
  add r9, 4
  dec ecx
  jnz @Scalar
@Done:
  vzeroupper
end;
// Native BF16 and INT4 projections also consume packed weights directly.
function BF16VectorDot(const Count: Integer; const Input: System.PSingle; const Weight: PBF16): Single; assembler;
asm
  .NOFRAME
  vxorps ymm0, ymm0, ymm0
  mov r10d, ecx
  shr r10d, 3
  jz @Reduce
@Loop:
  vpmovzxwd ymm1, [r8]
  vpslld ymm1, ymm1, 16
  vmulps ymm1, ymm1, [rdx]
  vaddps ymm0, ymm0, ymm1
  add r8, 16
  add rdx, 32
  dec r10d
  jnz @Loop
@Reduce:
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  and ecx, 7
  jz @Done
@Tail:
  movzx eax, word [r8]
  shl eax, 16
  vmovd xmm1, eax
  vmulss xmm1, xmm1, dword [rdx]
  vaddss xmm0, xmm0, xmm1
  add r8, 2
  add rdx, 4
  dec ecx
  jnz @Tail
@Done:
  vzeroupper
end;

function Int4VectorDot(const Count: Integer; const Input: System.PSingle; const Weight: PByte): Single; assembler;
asm
  .NOFRAME
  vxorps ymm0, ymm0, ymm0
  vmovdqu ymm3, [rip+INT4_SHIFTS]
  mov r10d, ecx
  shr r10d, 5
  jz @Reduce
@Group:
  vbroadcastss ymm4, dword [r8]
  vpbroadcastd ymm1, dword [r8+4]
  vpsrlvd ymm1, ymm1, ymm3
  vpand ymm1, ymm1, [rip+INT4_MASK]
  vpsubd ymm1, ymm1, [rip+INT4_CENTER]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpbroadcastd ymm1, dword [r8+8]
  vpsrlvd ymm1, ymm1, ymm3
  vpand ymm1, ymm1, [rip+INT4_MASK]
  vpsubd ymm1, ymm1, [rip+INT4_CENTER]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm0, ymm0, ymm1
  vpbroadcastd ymm1, dword [r8+12]
  vpsrlvd ymm1, ymm1, ymm3
  vpand ymm1, ymm1, [rip+INT4_MASK]
  vpsubd ymm1, ymm1, [rip+INT4_CENTER]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm0, ymm0, ymm1
  vpbroadcastd ymm1, dword [r8+16]
  vpsrlvd ymm1, ymm1, ymm3
  vpand ymm1, ymm1, [rip+INT4_MASK]
  vpsubd ymm1, ymm1, [rip+INT4_CENTER]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm4
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm0, ymm0, ymm1
  add r8, 20
  add rdx, 128
  dec r10d
  jnz @Group
@Reduce:
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  and ecx, 31
  jz @Done
  vbroadcastss ymm4, dword [r8]
  xor r11d, r11d
@Tail:
  mov eax, r11d
  shr eax, 1
  movzx eax, byte [r8+rax+4]
  test r11d, 1
  jz @Low
  shr eax, 4
@Low:
  and eax, 15
  sub eax, 8
  vcvtsi2ss xmm1, xmm1, eax
  vmulss xmm1, xmm1, xmm4
  vmulss xmm1, xmm1, dword [rdx]
  vaddss xmm0, xmm0, xmm1
  inc r11d
  add rdx, 4
  dec ecx
  jnz @Tail
@Done:
  vzeroupper
end;
{$ENDIF}
