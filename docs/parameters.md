# comfy_IrodoriTTS Parameters

この文書は、comfy_IrodoriTTSで提供される各ノード入力の説明です。

Irodori-TTS本体のCLI引数とは名前や分割単位が異なります。ComfyUIでは、基本生成に必要な入力を`IrodoriTTS Sampler`に置き、詳細設定は用途ごとのConfigノードとして接続します。

## 基本方針

- `IrodoriTTS Model Loader`でモデルと実行デバイスを選びます。
- `IrodoriTTS Sampler`でテキスト、秒数、ステップ数、seedを指定して生成します。
- 参照音声、VoiceDesign、CFG、Duration、Rescale、Schedule、Trim Tail、LoRAは必要な場合だけ追加します。
- Configノードを接続しない場合は、Sampler内部の標準値が使われます。

## IrodoriTTS Model Loader

IrodoriTTSのチェックポイントと実行設定をまとめ、`irodori_model_config`を出力します。

| Input | Default | 説明 |
| --- | --- | --- |
| `model` | checkpoint一覧 | 使用するIrodoriTTSチェックポイントです。ComfyUIの`models/checkpoints`に配置したファイルから選びます。 |
| `model_device` | 環境依存 | TTSモデルを実行するデバイスです。通常は`cuda`を使用します。 |
| `model_precision` | 環境依存 | TTSモデルの計算精度です。GPUでは`bf16`または`fp16`、互換性重視では`fp32`を選びます。 |
| `codec_device` | 環境依存 | codecを実行するデバイスです。VRAMを節約したい場合は`cpu`を選びます。 |
| `codec_precision` | 環境依存 | codecの計算精度です。互換性重視では`fp32`を選びます。 |
| `enable_watermark` | `False` | codec側のウォーターマーク処理を有効化します。通常は無効のままで構いません。 |
| `compile_model` | `False` | `torch.compile`でTTSモデルをコンパイルします。初回生成は遅くなりますが、環境によっては以後の生成が速くなります。 |
| `compile_dynamic` | `False` | `torch.compile`のdynamicモードを使います。入力長が変わる運用で試すための設定です。 |

チェックポイントの`latent_dim`に応じて、使用するcodecは内部で自動選択されます。

## IrodoriTTS Sampler

テキストから音声を生成し、ComfyUI標準の`AUDIO`を出力します。

### Required Inputs

| Input | Default | 説明 |
| --- | --- | --- |
| `model_config` | - | `IrodoriTTS Model Loader`の出力を接続します。 |
| `text` | - | 読み上げるテキストです。改行を含めた長文も指定できます。 |
| `seed` | `0` | 生成シードです。同じ条件で別候補を生成したい場合は値を変えます。 |
| `seconds` | `30.0` | 生成する音声長です。v3モデルでは`0`を指定すると自動秒数推定を使います。v1/v2モデルや推定器がない場合、`0`は30秒にフォールバックします。 |
| `num_steps` | `40` | サンプリングステップ数です。大きいほど時間がかかりますが、品質や安定性が変わる場合があります。 |

### Optional Config Inputs

| Input | 接続するノード | 説明 |
| --- | --- | --- |
| `lora_stack` | `IrodoriTTS LoRA Stack` | IrodoriTTS向けLoRAを適用します。 |
| `ref_config` | `IrodoriTTS Reference Audio` | 参照音声による話者・雰囲気の指定を行います。 |
| `voice_design_config` | `IrodoriTTS VoiceDesign Config` | VoiceDesignモデル向けの声質・話し方キャプションを指定します。 |
| `cfg_config` | `IrodoriTTS CFG Config` | テキスト条件、話者条件、適用時刻などのCFG詳細を指定します。 |
| `duration_config` | `IrodoriTTS Duration Config` | v3モデルの自動秒数推定を補正します。Samplerの`seconds = 0`時に意味があります。 |
| `rescale_config` | `IrodoriTTS Rescale Config` | 潜在の振れ幅やspeaker K/V補正を指定します。 |
| `schedule_config` | `IrodoriTTS Schedule Config` | RFサンプリングの時刻スケジュールを指定します。 |
| `trim_tail_config` | `IrodoriTTS Trim Tail Config` | 末尾切り詰め判定のしきい値を指定します。Samplerの`trim_tail`が有効なときに使われます。 |

### Other Inputs

| Input | Default | 説明 |
| --- | --- | --- |
| `batch_size` | `1` | 同一条件で生成する候補数です。大きくするとVRAM使用量が増えます。 |
| `decode_mode` | `sequential` | codecデコード方式です。`sequential`は省VRAM、`batch`は一括デコードです。 |
| `context_kv_cache` | `True` | コンテキストK/Vキャッシュを使います。通常は有効のままで構いません。 |
| `max_text_len` | `0` | テキストtoken長の上限です。`0`の場合はチェックポイント側の標準値を使います。 |
| `trim_tail` | `True` | 末尾の無音や平坦化した部分を切り詰めます。詳細判定は`trim_tail_config`で調整できます。 |

## IrodoriTTS Reference Audio

参照音声設定を作成し、Samplerの`ref_config`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `audio` | input一覧 | ComfyUIの`input`フォルダ内の音声または動画ファイルです。 |
| `normalize_ref_audio` | `False` | 参照音声を-16dB基準で正規化します。参照音声の音量差が大きい場合に有効です。 |
| `max_ref_seconds` | `30.0` | 参照として使う最大秒数です。長いファイルは先頭からこの秒数まで使われます。 |

動画ファイルを指定した場合は、ffmpegで音声を抽出します。`imageio-ffmpeg`またはシステムの`ffmpeg`が必要です。

VoiceDesignモデルでは参照音声ではなく`IrodoriTTS VoiceDesign Config`を使用します。

## IrodoriTTS VoiceDesign Config

VoiceDesignモデル向けのキャプション条件を作成し、Samplerの`voice_design_config`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `caption` | 空文字 | 声質、話速、感情、話し方などの説明文です。 |
| `cfg_scale_caption` | `3.0` | キャプション条件への追従強度です。上げるほどキャプションの影響が強くなります。 |
| `max_caption_len` | `0` | キャプションtoken長の上限です。`0`の場合はチェックポイント側の標準値を使います。 |

通常のIrodoriTTSモデルでは接続不要です。

## IrodoriTTS CFG Config

CFGの詳細設定を作成し、Samplerの`cfg_config`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `cfg_guidance_mode` | `independent` | CFGの適用方式です。`independent`、`joint`、`alternating`から選びます。 |
| `cfg_scale_text` | `3.0` | テキスト条件のCFG強度です。テキストへの追従が弱い場合に上げます。 |
| `cfg_scale_speaker` | `5.0` | 話者条件のCFG強度です。参照音声への追従が弱い場合に上げます。 |
| `cfg_scale_override` | `0.0` | テキスト・話者条件の共通CFG強度です。`0`の場合は無効です。 |
| `cfg_min_t` | `0.5` | CFGを適用する拡散時刻の下限です。 |
| `cfg_max_t` | `1.0` | CFGを適用する拡散時刻の上限です。 |

まずは未接続の標準値で生成し、必要な条件だけ調整するのがおすすめです。

## IrodoriTTS Duration Config

v3モデルの自動秒数推定を補正し、Samplerの`duration_config`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `duration_scale` | `1.0` | 自動推定された秒数に掛ける倍率です。長めにしたい場合は上げ、短めにしたい場合は下げます。 |
| `min_seconds` | `0.5` | 自動推定で許可する最短秒数です。 |
| `max_seconds` | `30.0` | 自動推定で許可する最長秒数です。 |

Samplerの`seconds`が`0`のときに使用されます。`seconds`に正の値を指定した場合、このConfigは生成秒数に影響しません。

## IrodoriTTS Rescale Config

潜在の振れ幅やspeaker K/V補正を指定し、Samplerの`rescale_config`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `truncation_factor` | `0.0` | 潜在の振れ幅を抑える係数です。`0`の場合は無効です。 |
| `rescale_k` | `0.0` | Rescale補正の強さです。`0`の場合は無効です。 |
| `rescale_sigma` | `0.0` | Rescale補正のsigmaです。`0`の場合は無効です。 |
| `speaker_kv_scale` | `0.0` | 話者条件K/Vの強調倍率です。`0`の場合は無効です。 |
| `speaker_kv_min_t` | `0.9` | speaker K/V補正を適用し始める拡散時刻です。 |
| `speaker_kv_max_layers` | `-1` | speaker K/V補正を適用する最大レイヤー数です。`-1`の場合は無効です。 |

音質や話者追従に強く影響するため、小さく変更しながら比較してください。

## IrodoriTTS Schedule Config

RFサンプリングの時刻スケジュールを指定し、Samplerの`schedule_config`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `schedule_mode` | `linear` | 時刻スケジュールです。`linear`または`sway`から選びます。 |
| `sway_coeff` | `-1.0` | `schedule_mode = sway`時の係数です。`linear`では使用されません。 |

未接続時は`linear`で生成します。通常は`linear`を基準にし、生成の安定性や質感を比較したい場合に`sway`を試します。

## IrodoriTTS Trim Tail Config

末尾切り詰め判定を調整し、Samplerの`trim_tail_config`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `tail_window_size` | `20` | 末尾判定に使う潜在窓サイズです。 |
| `tail_std_threshold` | `0.05` | 末尾が平坦化しているかを判定する標準偏差しきい値です。 |
| `tail_mean_threshold` | `0.1` | 末尾が平坦化しているかを判定する平均値しきい値です。 |

Samplerの`trim_tail`が有効なときに使用されます。未接続時は上記の標準値が使われます。

## IrodoriTTS LoRA Stack

IrodoriTTS向けLoRAをスタックし、Samplerの`lora_stack`へ接続します。

| Input | Default | 説明 |
| --- | --- | --- |
| `prev` | 未接続 | 前段の`IrodoriTTS LoRA Stack`出力です。複数LoRAを使う場合に接続します。 |
| `lora` | `None` | 追加するLoRAです。ComfyUIの`models/loras`から選びます。 |
| `strength` | `1.0` | LoRAの適用強度です。`1.0`が標準、`0.0`は実質無効です。 |

複数LoRAを使う場合は、前段の`irodori_lora_stack`を次の`prev`に接続して積み重ねます。

## IrodoriTTS Emoji Picker

IrodoriTTSで使いやすい絵文字を選ぶためのUIノードです。

生成処理に渡す出力はありません。テキスト入力へ絵文字を入力するための補助として使用します。

## Irodori Character Voice Sampler

Irodori Character Voice対応チェックポイントで、キャラクター画像を条件に音声を生成します。

既存の`IrodoriTTS Model Loader`の`irodori_model_config`を接続します。通常のIrodoriTTS Samplerとは別ノードで、参照音声やVoiceDesignキャプションは使用しません。

### Required Inputs

| Input | Default | 説明 |
| --- | --- | --- |
| `model_config` | - | `IrodoriTTS Model Loader`の出力を接続します。Character Voice対応チェックポイントを選択してください。 |
| `character_image` | - | 声質や話し方の条件に使うキャラクター画像です。バッチ画像の場合は先頭の1枚を使用します。 |
| `text` | - | 読み上げるテキストです。 |
| `seed` | `0` | 生成シードです。同じ条件で別候補を生成したい場合は値を変えます。 |
| `seconds` | `30.0` | 生成する音声長です。Character Voiceのデモ実装では30秒固定が標準です。 |
| `num_steps` | `40` | サンプリングステップ数です。 |
| `cfg_scale_character` | `3.0` | キャラクター画像条件のCFG強度です。上げるほど画像条件への追従が強くなります。 |

### Optional Config Inputs

| Input | 接続するノード | 説明 |
| --- | --- | --- |
| `cfg_config` | `IrodoriTTS CFG Config` | テキスト条件のCFG強度、CFG方式、適用時刻を指定します。`cfg_scale_speaker`は使用されません。 |
| `rescale_config` | `IrodoriTTS Rescale Config` | Rescale補正を指定します。speaker K/V補正は使用されません。 |
| `trim_tail_config` | `IrodoriTTS Trim Tail Config` | 末尾切り詰め判定のしきい値を指定します。Samplerの`trim_tail`が有効なときに使われます。 |

### Other Inputs

| Input | Default | 説明 |
| --- | --- | --- |
| `batch_size` | `1` | 同一条件で生成する候補数です。大きくするとVRAM使用量が増えます。 |
| `decode_mode` | `sequential` | codecデコード方式です。`sequential`は省VRAM、`batch`は一括デコードです。 |
| `context_kv_cache` | `True` | コンテキストK/Vキャッシュを使います。通常は有効のままで構いません。 |
| `max_text_len` | `0` | テキストtoken長の上限です。`0`の場合はチェックポイント側の標準値を使います。 |
| `trim_tail` | `True` | 末尾の無音や平坦化した部分を切り詰めます。 |
