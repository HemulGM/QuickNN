// CPU-prepared GGUF dots. Scales/biases are calculated once at model loading.
{$IF defined(CPUX64) and defined(USE_AVX2)}

const
  PREPARED_BYTE_15: array[0..31] of Byte = (15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15);
  PREPARED_BYTE_1: array[0..31] of Byte = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

function PreparedDotQ4K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer; const InputSums: System.PSingle): Single; assembler;
asm
  .SAVENV XMM8
  .SAVENV XMM7
  .PUSHNV R13
  .SAVENV XMM13
  .SAVENV XMM12
  mov r13, r9
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
@Block:
  xor r9d, r9d
@Group:
  vxorps ymm13, ymm13, ymm13
  vbroadcastss ymm2, dword [rcx+r9*4]
  vbroadcastss ymm3, dword [rcx+r9*4+32]
  mov r10d, r9d
  shr r10d, 1
  shl r10d, 5
  lea r10, [rcx+r10+64]
  mov eax, r9d
  and eax, 1
  shl eax, 2
  vmovd xmm5, eax
  vmovdqu ymm7, [r10]
  vpsrlw ymm7, ymm7, xmm5
  vpand ymm7, ymm7, [rip+PREPARED_BYTE_15]
  vpmovzxbd ymm1, xmm7
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm8, xmm7, 8
  vpmovzxbd ymm1, xmm8
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm13, ymm13, ymm1
  vextracti128 xmm8, ymm7, 1
  vpmovzxbd ymm1, xmm8
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm8, xmm8, 8
  vpmovzxbd ymm1, xmm8
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm13, ymm13, ymm1
  vmulps ymm13, ymm13, ymm2
  vmulps ymm4, ymm3, [r13]
  vsubps ymm13, ymm13, ymm4
  vaddps ymm0, ymm0, ymm13
  add r13, 32
  add rdx, 128
  inc r9d
  cmp r9d, 8
  jb @Group
  add rcx, 192
  dec r8d
  jnz @Block
  vaddps ymm13, ymm13, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PreparedDotQ5K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer; const InputSums: System.PSingle): Single; assembler;
asm
  .SAVENV XMM9
  .SAVENV XMM8
  .SAVENV XMM7
  .PUSHNV R13
  .SAVENV XMM13
  .SAVENV XMM11
  .SAVENV XMM12
  mov r13, r9
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
@Block:
  vmovdqu ymm9, [rcx+64]
  xor r9d, r9d
@Group:
  vxorps ymm13, ymm13, ymm13
  vbroadcastss ymm2, dword [rcx+r9*4]
  vbroadcastss ymm3, dword [rcx+r9*4+32]
  mov r10d, r9d
  shr r10d, 1
  shl r10d, 5
  lea r10, [rcx+r10+96]
  mov eax, r9d
  and eax, 1
  shl eax, 2
  vmovd xmm5, eax
  vmovd xmm11, r9d
  vmovdqu ymm7, [r10]
  vpsrlw ymm7, ymm7, xmm5
  vpand ymm7, ymm7, [rip+PREPARED_BYTE_15]
  vpsrlw ymm8, ymm9, xmm11
  vpand ymm8, ymm8, [rip+PREPARED_BYTE_1]
  vpsllw ymm8, ymm8, 4
  vpor ymm7, ymm7, ymm8
  vpmovzxbd ymm1, xmm7
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm8, xmm7, 8
  vpmovzxbd ymm1, xmm8
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm13, ymm13, ymm1
  vextracti128 xmm8, ymm7, 1
  vpmovzxbd ymm1, xmm8
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm8, xmm8, 8
  vpmovzxbd ymm1, xmm8
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm13, ymm13, ymm1
  vmulps ymm13, ymm13, ymm2
  vmulps ymm4, ymm3, [r13]
  vsubps ymm13, ymm13, ymm4
  vaddps ymm0, ymm0, ymm13
  add r13, 32
  add rdx, 128
  inc r9d
  cmp r9d, 8
  jb @Group
  add rcx, 224
  dec r8d
  jnz @Block
  vaddps ymm13, ymm13, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PreparedDotIQ4XS(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .SAVENV XMM13
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM10
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
  vmovdqu xmm6, [rip+DOT_IQ4]
  vpcmpeqb xmm10, xmm10, xmm10
  vpsrlw xmm10, xmm10, 12
  vpackuswb xmm10, xmm10, xmm10
@Block:
  xor r9d, r9d
@Group:
  vxorps ymm13, ymm13, ymm13
  vbroadcastss ymm2, dword [rcx+r9*4]
  mov r10d, r9d
  shl r10d, 4
  vmovdqu xmm5, [rcx+r10+32]
  vpand xmm4, xmm5, xmm10
  vpsrlw xmm5, xmm5, 4
  vpand xmm5, xmm5, xmm10
  vpshufb xmm4, xmm6, xmm4
  vpshufb xmm5, xmm6, xmm5
  vpmovsxbd ymm1, xmm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm4, xmm4, 8
  vpmovsxbd ymm1, xmm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm13, ymm13, ymm1
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm5, xmm5, 8
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm13, ymm13, ymm1
  vmulps ymm13, ymm13, ymm2
  vaddps ymm0, ymm0, ymm13
  add rdx, 128
  inc r9d
  cmp r9d, 8
  jb @Group
  add rcx, 160
  dec r8d
  jnz @Block
  vaddps ymm13, ymm13, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PreparedDotQ8(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .NOFRAME
  vxorps ymm0, ymm0, ymm0
@Block:
  vbroadcastss ymm2, dword [rcx]
  vpmovsxbd ymm1, qword [rcx+4]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm0, ymm0, ymm1
  vpmovsxbd ymm1, qword [rcx+12]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm0, ymm0, ymm1
  vpmovsxbd ymm1, qword [rcx+20]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm0, ymm0, ymm1
  vpmovsxbd ymm1, qword [rcx+28]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, ymm2
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm0, ymm0, ymm1
  add rcx, 36
  add rdx, 128
  dec r8d
  jnz @Block
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PreparedDotIQ4NL(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .SAVENV XMM13
  vxorps ymm0, ymm0, ymm0
@Block:
  vxorps ymm13, ymm13, ymm13
  vbroadcastss ymm2, dword [rcx]
  vmovdqu xmm3, [rip+DOT_IQ4]
  vmovdqu xmm5, [rcx+4]
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
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm4, xmm4, 8
  vpmovsxbd ymm1, xmm4
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm13, ymm13, ymm1
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm13, ymm13, ymm1
  vpsrldq xmm5, xmm5, 8
  vpmovsxbd ymm1, xmm5
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm13, ymm13, ymm1
  vmulps ymm13, ymm13, ymm2
  vaddps ymm0, ymm0, ymm13
  add rcx, 20
  add rdx, 128
  dec r8d
  jnz @Block
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PreparedDotQ6K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM11
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
  mov eax, 32
  vmovd xmm6, eax
  vpbroadcastd ymm6, xmm6
@Block:
  xor r9d, r9d
@Group:
  vbroadcastss ymm2, dword [rcx+r9*4]
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
  lea r10, [rcx+r10+64]
  mov r11d, r9d
  shr r11d, 3
  shl r11d, 5
  add r11d, eax
  lea r11, [rcx+r11+192]
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
  add rcx, 256
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PreparedDotQ3K(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .PUSHNV RBX
  .SAVENV XMM6
  .SAVENV XMM11
  .SAVENV XMM12
  vxorps ymm0, ymm0, ymm0
  vxorps ymm12, ymm12, ymm12
  mov eax, 4
  vmovd xmm6, eax
  vpbroadcastd ymm6, xmm6
@Block:
  xor r9d, r9d
@Group:
  vbroadcastss ymm2, dword [rcx+r9*4]
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
  lea r10, [rcx+r10+96]
  lea r11, [rcx+rax+64]
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
  add rcx, 160
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm12
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;

function PreparedDotIQ3S(const Data: PByte; const Src: System.PSingle; const Blocks: Integer): Single; assembler;
asm
  .SAVENV XMM13
  vxorps ymm0, ymm0, ymm0
  vxorps ymm5, ymm5, ymm5
@Block:
  xor r9d, r9d
@Group:
  vxorps ymm13, ymm13, ymm13
  vbroadcastss ymm2, dword [rcx+r9*4]
  mov r10d, r9d
  shl r10d, 5
  lea r10, [rcx+r10+32]
  vpmovsxbd ymm1, qword [r10+0]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+0]
  vaddps ymm13, ymm13, ymm1
  vpmovsxbd ymm1, qword [r10+8]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+32]
  vaddps ymm13, ymm13, ymm1
  vpmovsxbd ymm1, qword [r10+16]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+64]
  vaddps ymm13, ymm13, ymm1
  vpmovsxbd ymm1, qword [r10+24]
  vcvtdq2ps ymm1, ymm1
  vmulps ymm1, ymm1, [rdx+96]
  vaddps ymm13, ymm13, ymm1
  vmulps ymm13, ymm13, ymm2
  vaddps ymm0, ymm0, ymm13
  add rdx, 128
  inc r9d
  cmp r9d, 8
  jb @Group
  add rcx, 288
  dec r8d
  jnz @Block
  vaddps ymm0, ymm0, ymm5
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
end;
procedure PreparedInputSums(const Input, Output: System.PSingle; const Groups: Integer); assembler;
asm
  .NOFRAME
@Loop:
  vmovups ymm0, [rcx]
  vaddps ymm0, ymm0, [rcx+32]
  vmovups ymm1, [rcx+64]
  vaddps ymm1, ymm1, [rcx+96]
  vaddps ymm0, ymm0, ymm1
  vmovups [rdx], ymm0
  add rcx, 128
  add rdx, 32
  dec r8d
  jnz @Loop
  vzeroupper
end;
{$ENDIF}
