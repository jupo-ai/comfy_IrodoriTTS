import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CLASS_NAMES = [
    "IrodoriTTSReferenceAudio"
];

const ACCEPT_EXTS = [
    "audio/mpeg", 
    "audio/wav", 
    "audio/x-wav", 
    "audio/ogg", 
]

const extension = {
    name: "IrodoriTTSReferenceAudio", 

    beforeRegisterNodeDef: function(nodeType, nodeData, app) {
        if (!CLASS_NAMES.includes(nodeType.comfyClass)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const widgetName = "ref_audio"; // Python側のINPUT_TYPESのキー名
            const widget = this.widgets.find((w) => w.name === widgetName);

            // --- 1. オーディオプレイヤー（HTML要素）の作成 ---
            const audioEl = document.createElement("audio");
            audioEl.controls = true; // コントロール（再生/停止/シーク）を表示
            audioEl.style.width = "100%";
            audioEl.style.marginTop = "4px";

            // オーディオのソースを更新するヘルパー関数
            const updateAudioSrc = (filename) => {
                if (!filename) return;
                
                // プレビュー用のURLを生成 (/view APIを使用)
                // type=input は inputフォルダを参照します
                const params = new URLSearchParams({
                    filename: filename,
                    type: "input",
                });
                
                // api.apiURLを使って完全なURLを取得
                const url = api.apiURL("/view?" + params.toString());
                
                // 同じファイルでなければ更新
                if (audioEl.src !== url) {
                    audioEl.src = url;
                }
            };

            // --- 2. ノードにDOMウィジェットとして追加 ---
            // addDOMWidget(名前, タイプ, 要素, オプション)
            this.addDOMWidget("audio_preview", "audio", audioEl, {
                serialize: false, // 保存不要
                hideOnZoom: false 
            });


            // --- 3. 既存のコンボボックス変更時にプレイヤーを更新する ---
            const originalCallback = widget.callback;
            widget.callback = (v) => {
                updateAudioSrc(v);
                if (originalCallback) {
                    originalCallback(v);
                }
            };
            
            // 初期値が入っている場合も更新
            if (widget.value) {
                updateAudioSrc(widget.value);
            }


            // --- 4. アップロードボタンの処理 ---
            const uploadWidget = this.addWidget("button", "choose file to upload", "audio", () => {
                const fileInput = document.createElement("input");
                Object.assign(fileInput, {
                    type: "file",
                    accept: ACCEPT_EXTS.join(","),
                    style: "display: none",
                    
                    onchange: async () => {
                        if (fileInput.files.length) {
                            const file = fileInput.files[0];
                            const formData = new FormData();
                            
                            formData.append("image", file); // API仕様によりキーはimage
                            formData.append("overwrite", "true");
                            formData.append("type", "input");

                            try {
                                const resp = await api.fetchApi("/upload/image", {
                                    method: "POST",
                                    body: formData,
                                });

                                if (resp.status === 200) {
                                    const data = await resp.json();
                                    const filename = data.name;

                                    if (widget) {
                                        if (!widget.options.values.includes(filename)) {
                                            widget.options.values.push(filename);
                                        }
                                        widget.value = filename;
                                        
                                        // 値の変更を通知し、プレイヤーも更新
                                        if (widget.callback) {
                                            widget.callback(widget.value);
                                        }
                                    }
                                } else {
                                    alert("Upload failed: " + resp.statusText);
                                }
                            } catch (error) {
                                console.error(error);
                                alert("Upload failed");
                            }
                        }
                    },
                });

                document.body.appendChild(fileInput);
                fileInput.click();
                
                setTimeout(() => {
                    fileInput.remove();
                }, 100);
            });
            
            uploadWidget.serialize = false;

            // ノードのサイズを少し広げてプレイヤーが見えやすくする
            this.setSize([this.size[0], this.size[1] + 40]);

            return r;
        };
    }, 
};

app.registerExtension(extension);