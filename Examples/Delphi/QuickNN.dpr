program QuickNN;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  System.Generics.Collections,
  QuickNN.SafeTensors,
  QuickNN.Kernels,
  QuickNN.Vulkan,
  QuickNN.Common,
  QuickNN.Tokenizer,
  QuickNN.Models.Transformers,
  QuickNN.Models.VAE,
  QuickNN.Sampling,
  QuickNN.Models.Qwen3,
  QuickNN.Models.Flux,
  QuickNN.Models.ZImage,
  QuickNN.Console.Tensors,
  QuickNN.Timing,
  QuickNN.Console.Ansi;

var
  Dict: TDictionary<string, PMemoryBlock>;
  Flux: TQNNFlux;
  Params: TGenerateParams;
  Img: TQNNImage;
  ImgFile: string;
  T, MemAllocated: Int64;

{$STACKFRAMES on}

procedure AfterblockForward(const Typ: TSubstepType; const I, Total: LongInt);
begin
  case Typ of
    SUBSTEP_DOUBLE_BLOCK: Write('D');
    SUBSTEP_SINGLE_BLOCK:
      if I mod 5 = 0 then
        Write('S');
    SUBSTEP_FINAL_LAYER: WriteLn('F [', ((Clock - T) / CLOCKS_PER_SEC): 1: 3, '] Seconds');
  end;

  if I = 0 then
    T := Clock();
end;

procedure Afterstep(const I, Total: LongInt);
begin
  // if i=total-1 then
  //  write(#13' ',i+1,'/',total, ' ')
  // else
  if I = 0 then
  begin
    Write(#13#10' ', I + 1, '/', Total, ' ')
  end
  else
    Write(#13' ', I + 1, '/', Total, ' ')
end;

procedure Afterphase(const Status: string; const Done: Boolean);
begin
  if Done then
  begin
    WriteLn(' Finished [', ((Clock - T) / CLOCKS_PER_SEC): 1: 3, '] Seconds')
  end
  else
  begin
    T := Clock();
    Write(Status, '...', TAB)
  end;
end;

procedure DEBUGAllMems();
var
  M: TPair<string, PMemoryBlock>;
  Sum: IntPtr;
begin
  Sum := 0;
  for M in Dict do
  begin

    /// /if m.value.name<>'' then
    WriteLn(M.Value.Name, ' = ', M.Value.Size * DATATYPE_BITS[M.Value.DataType] div 8);
    Inc(Sum, M.Value.Size * DATATYPE_BITS[M.Value.DataType] div 8)
  end;
  WriteLn('SUM MEM =', Sum, ' ALLOCATED =', MemAllocated);
  if Sum <> MemAllocated then
  begin
    // SysUtils.beep;
    ReadLn;
  end;
end;

procedure MemoryChange(const Status: string; const Old, New: IntPtr; const Mem: TMemoryBlock);
var
  Delta: IntPtr;
begin
  Assert(Trim(Mem.Name) <> '', 'Memory has no name tag');
  SaveCursorPos();
  Delta := New - Old;
  if not ((New > 0) and (Old > 0)) then
  begin
    if not Dict.ContainsKey(Mem.Name) then
    begin
      Assert(Status = 'new');
      Dict.Add(Mem.Name, @Mem);
    end;
    if New = 0 then
    begin
      Assert(Dict.ContainsKey(Mem.Name));
      Dict.Remove(Mem.Name);
    end;
  end;
  // s := dict[mem.name].size;
  if (MemAllocated + Delta) < 0 then
    ReadLn;
  Inc(MemAllocated, Delta);
  // DEBUGAllMems();
  CursorAbsPos(100, 1);
  Write(MemAllocated / 1000000000: 1: 3, ' GB');
  RestoreCursorPos();
end;

var
  // l_w, l_h: longint;
  // l: TMemoryBlock;
  Prompt: string;

begin
  var GPUStatus := QNNGetGPUStatus;
  if GPUStatus.Available then
    WriteLn('Matrix backend: Vulkan on ', GPUStatus.DeviceName, ' (large GEMM and BF16 Linear)')
  else
  begin
    WriteLn('Matrix backend: CPU');
    if GPUStatus.LastError <> '' then
      WriteLn('Vulkan: ', GPUStatus.LastError);
  end;
  Dict := TDictionary<string, PMemoryBlock>.Create();
  OnMemoryUpdate := MemoryChange;
  Params := Default(TGenerateParams);
  Params.Width := 512;
  Params.Height := 512;
  // params.seed:= 666;
  // params.seed:=1781898218;
  Params.PowerAlpha := 2;
  Params.NumSteps := 4;
  SubstepCallback := AfterblockForward;
  StepCallback := Afterstep;
  TextProgressCallback := Afterstep;
  VAEProgressCallback := Afterstep;
  // PROMPT := 'a cute realistic panda holding a "I will code for food!" signboard';
  Prompt :=
      'standing facing an old stone wall blue flowers, (full length: 1.5) photo of a 30 years old (naked: 1.5) fully nude body (godess:1.2) with flowing long (messy:1.1) radiant (blonde:1.2) hair, (backlit:1.1), analog color film photograph, 50mm F1.2L lens (shallow:1.2) depth of field, emotions, sensual pose, high dynamic range, HDR, low key, (subsurface scattering:1.2), (moody:0.9) shadows and (complex:1.1) lighting. (Green:0.5) radiant eyes, RAW photography. she totally (naked: 1.5). revealing her naked ass, (back:1.2) camera, dynamic (volumetric:0.9) lighting, (godrays:0.8), best quality, masterpiece, highres, ultra-detailed, girl (wide:1.2) hips. .The light is coming from above her. totally naked. nude body. nfsw. naked ass. full body';
  {$DEFINE _ZIMAGE}
{$IFDEF ZIMAGE}
  var ZImage := TQNNZImage.Load('../../Z-Image-Turbo', Afterphase);
  ZImage.UseMmap := True;
  Img := ZImage.Generate(UTF8String(Prompt), Params);
  // img := zimage.generate('a realistic pink grisly bear wearing a blue hat holding a (I''m Sad) signboard.', params);
  // img := zimage.generate('On a desk, a laptop displays a wallpaper of a countryside landscape with a small river. The landscape extends beyond the computer screen, invading the desk, the walls, and the room. The river overflows the screen, and plants and trees grow beyond its boundaries, blending seamlessly with the surrounding environment. The style is a stunning 3D rendering, with deep, cinematic lighting.', params);
  // img := zimage.generate('a cute pink raccoon holding a "I''m Sad" signboard', params);
  // img := zimage.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  // img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
  // img := flux.generate('أرنب روبوتي يغني في الفضاء', params);
  // img := flux.generate('أرنب روبوتي يتناول العشاء مع قطة', params);
  // img := flux.generate('تفاحة', params);
  ZImage.Free;
{$ELSE}

  // var src := TQNNImage.loadFromFile('C:\development\flux2.c\bear.png');
  // src.print();
  // printSixel(pointer(src.data), src.width, src.height, true);
  // flux := TQNNFLux.load('c:\development\flux2.c\FLUX.2-klein-9B');
  // flux := TQNNFlux.load('../../../models/' + 'FLUX.2-klein-4B', afterphase);
  Flux := TQNNFlux.Load('E:\Models\FLUX.2-klein-9B', Afterphase);

  Flux.UseMmap := True;
  // flux.loadVAE('c:\development\flux2.c\flux-klein-model');
  // l := flux.vae.encode(src.asMemoryBlock(), 1, src.height, src.width, l_h, l_w);
  // img := flux.vae.decode(l, 1, l_h, l_w);
  // img.print;
  //  readln;
  // img := flux.generate('a realistic pink grisly bear wearing a blue hat holding (will code for food) signboard.', params);
  // img := flux.generate('On a desk, a laptop displays a wallpaper of a countryside landscape with a small river. The landscape extends beyond the computer screen, invading the desk, the walls, and the room. The river overflows the screen, and plants and trees grow beyond its boundaries, blending seamlessly with the surrounding environment. The style is a stunning 3D rendering, with deep, cinematic lighting.', params);
  for var I := 1 to 1 do
  begin
    // PROMPT := 'european woman, (28.5) year old, naked, without clothes, without panties, full body, Olive warm skin tone, perfect boobs,(NSFW) , (masterpiece) , pretty tits, ((nipples) ), light nipples, cute nipples, normal nipples, no clothes, naked, (wet wavy hair),(slutty face) , (beautiful eyes:0.944) , (sexy) ,  detailed vagina, (vagina:1.1), (good pussy), spreading legs open pussy, ((good pussy) ), (juicy pussy) , pink pussy, (open pussy) , (pussy hair) , vagina, camel toe, ((clit) ), no underwear, extremely detailed, face detail, face retouch, polish, slutty, pink nipple, thick, curvy, voluptuous hourglass figure, (naked:1.1), highly detailed face, symmetrical full breasts, natural droop, medium-sized areolas, smooth transitions, subtle pectoralis definition, minimally visible sternum, slight abdominal convexity, gleaming lips,beneath soft fat layer, visible cleavage , wide pelvic
    // region (45-50 inches), soft inguinal lines, without clothes, anatomically accurate pubic region, graceful, natural proportions, ultra-detailed, medically precise, high resolution, realistic textures, without panties, smooth skin, soft lighting, professional anatomical context, (sensual:1.4), (glistening:1.2). The scene is shot with (Kodakportra1600:1.1), (filmgrain:1.1), (Analogcamera:1.1), (Sharpdetails:1.1), (50mm:1.1).';
    // PROMPT := 'Hot nude girl, full body view, high quality, long hair, pussy, genetalia, big breast, sweat::4.8 da vinci::3 galactic::3 blinding light::3 360 panorama::3 picasso::3 indigo::3 nickel::3 nylon::3  --aspect 16:9 --version 4 --quality 1';
    // PROMPT := 'A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed';
    Prompt :=
        'Cinematic high-resolution vertical photo in street style, soft natural lighting on an overcast day without harsh shadows. Hip plan (American plan), foreshortening at eye level. In the center of the frame is a broad—shouldered, muscular, athletic man in a relaxed pose. The face is turned slightly down, the eyes are hidden behind dark "droplet" sunglasses with white frames. The hero is dressed in a gray vintage leather jacket with a pronounced scuffed texture and silver zippers, a white T-shirt under the jacket, and light blue straight-cut jeans underneath. His left hand is casually tucked into his jeans pocket. In his right hand, he holds a giant, heavy and dense bouquet of dark red roses, which, due to its huge volume, occupies a significant part of the left half of the frame and covers '
            + 'part of the jacket. The hero stands close to a dark green retro sports car with a low seat; part of the roof, curved rear window and door are visible, raindrops on the glossy body. The background is full of details: a black wrought-iron fence with gold decorative peaks, behind the fence you can see the facade of a red brick building with massive white columns and a window. Deep sharpness on the hero, the background is slightly blurred. Mandatory: 100% preserve the identity of the face and proportions of the person from the photo reference, the natural texture of the skin. Prohibited: changing facial features, plastic skin, deforming hands and fingers, extra limbs, random people, watermarks, text, logos, branding, curved lines, impaired perspective, hanging objects, unnatural glare, lack '
            + 'of reflections.';
    Params.Seed := Random(Integer.MaxValue);

    Img := Flux.Generate(UTF8String(Prompt), Params);
    // img := flux.generate('a young lady with sunglasses, red hair and little frickles, nude', params);
    // img := flux.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
    // img := flux.generate('Tom and Jerry', params);
    // var imgs: TArray<TQNNImage>;
    // setLength(imgs, 2);
    // imgs[0] := TQNNImage.loadFromFile('c:\development\flux2.c\car.jpg');
    // imgs[1] := TQNNImage.loadFromFile('c:\development\flux2.c\beach.jpg');

    // img := flux.generate('A pink grizly bear wearing a blue fedora in a cozy room with green furniture and holding a sign with "I will code for  food!" signboard.', params);
    // img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
    // img := flux.generate('أرنب روبوتي يغني في الفضاء', params);
    // img := flux.generate('أرنب روبوتي يتناول العشاء مع قطة', params);
    // img := flux.generate('تفاحة', params);
{$ENDIF}
    ImgFile := GetCurrentDir() + PathDelim + FormatDateTime('YYYY_MM_DD_hhnnsszzz', Now()) + '_pascal.png';
    WriteLn('Saving to [', ImgFile, ']');
    Img.SaveToFile(ImgFile);
    Img.PrintSixel;
    Img.Free;
  end;
  Flux.Free;
  // DEBUGAllMems();
  //  freeAndNil(dict);
  ReadLn;
end.
