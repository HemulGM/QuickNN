// Fused packed-weight / FP32 activation dots; no expanded matrix or row buffer.
{$IF defined(CPUX64) and defined(USE_AVX2)}
const
  DOT_MASK_3: array[0..7] of Integer = (3,3,3,3,3,3,3,3);
  DOT_MASK_1: array[0..7] of Integer = (1,1,1,1,1,1,1,1);
  DOT_MASK_15: array[0..7] of Integer = (15,15,15,15,15,15,15,15);
  DOT_IQ4: array[0..15] of ShortInt = (-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);

function PackedDotQ4K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM7
  .SAVENV XMM8
  .SAVENV XMM9
  .SAVENV XMM10
  .SAVENV XMM11
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
@Block:
  vmovd xmm7, dword [rcx]
  vcvtph2ps xmm7, xmm7
  vbroadcastss ymm8, xmm7
  vshufps xmm9, xmm7, xmm7, 85
  vbroadcastss ymm9, xmm9
  xor r9d, r9d
@Group:
  cmp r9d, 4
  jae @HighScale
  movzx eax, byte [rcx+r9+4]
  and eax, 63
  movzx ebx, byte [rcx+r9+8]
  and ebx, 63
  jmp @ScaleReady
@HighScale:
  movzx eax, byte [rcx+r9]
  shr eax, 6
  shl eax, 4
  movzx r11d, byte [rcx+r9+8]
  and r11d, 15
  or eax, r11d
  movzx ebx, byte [rcx+r9+4]
  shr ebx, 6
  shl ebx, 4
  movzx r11d, byte [rcx+r9+8]
  shr r11d, 4
  or ebx, r11d
@ScaleReady:
  vxorps xmm2, xmm2, xmm2
  vcvtsi2ss xmm2, xmm2, eax
  vbroadcastss ymm2, xmm2
  vmulps ymm2, ymm2, ymm8
  vxorps xmm3, xmm3, xmm3
  vcvtsi2ss xmm3, xmm3, ebx
  vbroadcastss ymm3, xmm3
  vmulps ymm3, ymm3, ymm9
  mov r10d, r9d
  shr r10d, 1
  shl r10d, 5
  lea r10, [rcx+r10+16]
  mov eax, r9d
  and eax, 1
  shl eax, 2
  vmovd xmm5, eax
  vpmovzxbd ymm1, qword [r10+0]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpmovzxbd ymm1, qword [r10+8]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm12, ymm12, ymm1
  vpmovzxbd ymm1, qword [r10+16]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm0, ymm0, ymm1
  vpmovzxbd ymm1, qword [r10+24]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm12, ymm12, ymm1
  add rdx, 128
  inc r9d
  cmp r9d, 8
  jb @Group
  add rcx, 144
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PackedDotQ5K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM7
  .SAVENV XMM8
  .SAVENV XMM9
  .SAVENV XMM10
  .SAVENV XMM11
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
@Block:
  vmovd xmm7, dword [rcx]
  vcvtph2ps xmm7, xmm7
  vbroadcastss ymm8, xmm7
  vshufps xmm9, xmm7, xmm7, 85
  vbroadcastss ymm9, xmm9
  xor r9d, r9d
@Group:
  cmp r9d, 4
  jae @HighScale
  movzx eax, byte [rcx+r9+4]
  and eax, 63
  movzx ebx, byte [rcx+r9+8]
  and ebx, 63
  jmp @ScaleReady
@HighScale:
  movzx eax, byte [rcx+r9]
  shr eax, 6
  shl eax, 4
  movzx r11d, byte [rcx+r9+8]
  and r11d, 15
  or eax, r11d
  movzx ebx, byte [rcx+r9+4]
  shr ebx, 6
  shl ebx, 4
  movzx r11d, byte [rcx+r9+8]
  shr r11d, 4
  or ebx, r11d
@ScaleReady:
  vxorps xmm2, xmm2, xmm2
  vcvtsi2ss xmm2, xmm2, eax
  vbroadcastss ymm2, xmm2
  vmulps ymm2, ymm2, ymm8
  vxorps xmm3, xmm3, xmm3
  vcvtsi2ss xmm3, xmm3, ebx
  vbroadcastss ymm3, xmm3
  vmulps ymm3, ymm3, ymm9
  mov r10d, r9d
  shr r10d, 1
  shl r10d, 5
  lea r10, [rcx+r10+48]
  mov eax, r9d
  and eax, 1
  shl eax, 2
  vmovd xmm5, eax
  vmovd xmm11, r9d
  vpmovzxbd ymm1, qword [r10+0]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vpmovzxbd ymm4, qword [rcx+16]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_1]
  vpslld ymm4, ymm4, 4
  vpor ymm1, ymm1, ymm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpmovzxbd ymm1, qword [r10+8]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vpmovzxbd ymm4, qword [rcx+24]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_1]
  vpslld ymm4, ymm4, 4
  vpor ymm1, ymm1, ymm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm12, ymm12, ymm1
  vpmovzxbd ymm1, qword [r10+16]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vpmovzxbd ymm4, qword [rcx+32]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_1]
  vpslld ymm4, ymm4, 4
  vpor ymm1, ymm1, ymm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm0, ymm0, ymm1
  vpmovzxbd ymm1, qword [r10+24]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vpmovzxbd ymm4, qword [rcx+40]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_1]
  vpslld ymm4, ymm4, 4
  vpor ymm1, ymm1, ymm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vsubps ymm1, ymm1, ymm3
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm12, ymm12, ymm1
  add rdx, 128
  inc r9d
  cmp r9d, 8
  jb @Group
  add rcx, 176
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PackedDotIQ4XS(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM7
  .SAVENV XMM8
  .SAVENV XMM9
  .SAVENV XMM10
  .SAVENV XMM11
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
  vmovdqu xmm6, [rip+DOT_IQ4]
  vpcmpeqb xmm10, xmm10, xmm10
  vpsrlw xmm10, xmm10, 12
  vpackuswb xmm10, xmm10, xmm10
@Block:
  movzx eax, word [rcx]
  vmovd xmm7, eax
  vcvtph2ps xmm7, xmm7
  vbroadcastss ymm8, xmm7
  xor r9d, r9d
@Group:
  mov eax, r9d
  shr eax, 1
  movzx ebx, byte [rcx+rax+4]
  mov eax, r9d
  and eax, 1
  shl eax, 2
  mov r10, rcx
  mov ecx, eax
  shr ebx, cl
  and ebx, 15
  movzx eax, word [r10+2]
  mov ecx, r9d
  shl ecx, 1
  shr eax, cl
  and eax, 3
  shl eax, 4
  or eax, ebx
  sub eax, 32
  mov rcx, r10
  vxorps xmm2, xmm2, xmm2
  vcvtsi2ss xmm2, xmm2, eax
  vbroadcastss ymm2, xmm2
  vmulps ymm2, ymm2, ymm8
  mov r10d, r9d
  shl r10d, 4
  vmovdqu xmm5, [rcx+r10+8]
  vpand xmm4, xmm5, xmm10
  vpsrlw xmm5, xmm5, 4
  vpand xmm5, xmm5, xmm10
  vpshufb xmm4, xmm6, xmm4
  vpshufb xmm5, xmm6, xmm5
  vpmovsxbd ymm1, xmm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpsrldq xmm4, xmm4, 8
  vpmovsxbd ymm1, xmm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm12, ymm12, ymm1
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm0, ymm0, ymm1
  vpsrldq xmm5, xmm5, 8
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm12, ymm12, ymm1
  add rdx, 128
  inc r9d
  cmp r9d, 8
  jb @Group
  add rcx, 136
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;


function PackedDotQ8(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .NOFRAME
  vxorps ymm0, ymm0, ymm0
@Block:
  movzx eax, word [rcx]
  vmovd xmm2, eax
  vcvtph2ps xmm2, xmm2
  vbroadcastss ymm2, xmm2
  vpmovsxbd ymm1, qword [rcx+2]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpmovsxbd ymm1, qword [rcx+10]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm0, ymm0, ymm1
  vpmovsxbd ymm1, qword [rcx+18]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm0, ymm0, ymm1
  vpmovsxbd ymm1, qword [rcx+26]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm0, ymm0, ymm1
  add rcx, 34
  add rdx, 128
  dec r8d
  jnz @Block
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PackedDotIQ4NL(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .NOFRAME
  vxorps ymm0, ymm0, ymm0
@Block:
  movzx eax, word [rcx]
  vmovd xmm2, eax
  vcvtph2ps xmm2, xmm2
  vbroadcastss ymm2, xmm2
  vmovdqu xmm3, [rip+DOT_IQ4]
  vmovdqu xmm5, [rcx+2]
  vpcmpeqb xmm1, xmm1, xmm1
  vpsrlw xmm1, xmm1, 12
  vpackuswb xmm1, xmm1, xmm1
  vpand xmm4, xmm5, xmm1
  vpsrlw xmm5, xmm5, 4
  vpand xmm5, xmm5, xmm1
  vpshufb xmm4, xmm3, xmm4
  vpshufb xmm5, xmm3, xmm5
  vpmovsxbd ymm1, xmm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpsrldq xmm4, xmm4, 8
  vpmovsxbd ymm1, xmm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm0, ymm0, ymm1
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm0, ymm0, ymm1
  vpsrldq xmm5, xmm5, 8
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm0, ymm0, ymm1
  add rcx, 18
  add rdx, 128
  dec r8d
  jnz @Block
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;


function PackedDotQ6K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM8
  .SAVENV XMM11
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
  mov eax, 32
  vmovd xmm6, eax
  vpbroadcastd ymm6, xmm6
@Block:
  movzx eax, word [rcx+208]
  vmovd xmm2, eax
  vcvtph2ps xmm2, xmm2
  vbroadcastss ymm8, xmm2
  xor r9d, r9d
@Group:
  movsx eax, byte [rcx+r9+192]
  vxorps xmm2, xmm2, xmm2
  vcvtsi2ss xmm2, xmm2, eax
  vbroadcastss ymm2, xmm2
  vmulps ymm2, ymm2, ymm8
  mov eax, r9d
  shr eax, 1
  and eax, 3
  mov ebx, eax
  shr eax, 1
  shl eax, 2
  vmovd xmm5, eax
  shl ebx, 1
  vmovd xmm11, ebx
  mov r10d, r9d
  shr r10d, 3
  shl r10d, 6
  mov eax, r9d
  shr eax, 1
  and eax, 1
  shl eax, 5
  add r10d, eax
  mov eax, r9d
  and eax, 1
  shl eax, 4
  add r10d, eax
  add r10, rcx
  mov r11d, r9d
  shr r11d, 3
  shl r11d, 5
  add r11d, eax
  lea r11, [rcx+r11+128]
  vpmovzxbd ymm1, qword [r10+0]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vpmovzxbd ymm4, qword [r11+0]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_3]
  vpslld ymm4, ymm4, 4
  vpor ymm1, ymm1, ymm4
  vpsubd ymm1, ymm1, ymm6
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpmovzxbd ymm1, qword [r10+8]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_15]
  vpmovzxbd ymm4, qword [r11+8]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_3]
  vpslld ymm4, ymm4, 4
  vpor ymm1, ymm1, ymm4
  vpsubd ymm1, ymm1, ymm6
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm12, ymm12, ymm1
  add rdx, 64
  inc r9d
  cmp r9d, 16
  jb @Group
  add rcx, 210
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PackedDotQ3K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM8
  .SAVENV XMM11
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
  mov eax, 4
  vmovd xmm6, eax
  vpbroadcastd ymm6, xmm6
@Block:
  movzx eax, word [rcx+108]
  vmovd xmm2, eax
  vcvtph2ps xmm2, xmm2
  vbroadcastss ymm8, xmm2
  xor r9d, r9d
@Group:
  mov r10, rcx
  mov eax, r9d
  and eax, 7
  movzx ebx, byte [r10+rax+96]
  mov ecx, r9d
  shr ecx, 3
  shl ecx, 2
  shr ebx, cl
  and ebx, 15
  mov eax, r9d
  and eax, 3
  movzx eax, byte [r10+rax+104]
  mov ecx, r9d
  shr ecx, 2
  shl ecx, 1
  shr eax, cl
  and eax, 3
  shl eax, 4
  or eax, ebx
  sub eax, 32
  mov rcx, r10
  vxorps xmm2, xmm2, xmm2
  vcvtsi2ss xmm2, xmm2, eax
  vbroadcastss ymm2, xmm2
  vmulps ymm2, ymm2, ymm8
  mov eax, r9d
  shr eax, 1
  vmovd xmm11, eax
  and eax, 3
  shl eax, 1
  vmovd xmm5, eax
  mov r10d, r9d
  shr r10d, 3
  shl r10d, 5
  mov eax, r9d
  and eax, 1
  shl eax, 4
  add r10d, eax
  lea r10, [rcx+r10+32]
  lea r11, [rcx+rax]
  vpmovzxbd ymm1, qword [r10+0]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_3]
  vpmovzxbd ymm4, qword [r11+0]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_1]
  vpslld ymm4, ymm4, 2
  vpor ymm1, ymm1, ymm4
  vpsubd ymm1, ymm1, ymm6
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpmovzxbd ymm1, qword [r10+8]
  vpsrld ymm1, ymm1, xmm5
  vpand ymm1, ymm1, [rip+DOT_MASK_3]
  vpmovzxbd ymm4, qword [r11+8]
  vpsrld ymm4, ymm4, xmm11
  vpand ymm4, ymm4, [rip+DOT_MASK_1]
  vpslld ymm4, ymm4, 2
  vpor ymm1, ymm1, ymm4
  vpsubd ymm1, ymm1, ymm6
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm12, ymm12, ymm1
  add rdx, 64
  inc r9d
  cmp r9d, 16
  jb @Group
  add rcx, 110
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

{$ENDIF}
