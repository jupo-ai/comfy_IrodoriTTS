from typing_extensions import override
from comfy_api.latest import ComfyExtension

from . import nodes

class Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            nodes.IrodoriTTSModelLoader, 
            nodes.IrodoriTTSModelLoaderHF, 
            nodes.IrodoriTTSReferenceAudio, 
            nodes.IrodoriTTSAdvancedCFG, 
            nodes.IrodoriTTSRescaleConfig, 
            nodes.IrodoriTTSTailConfig,
            nodes.IrodoriTTSSampler, 
            nodes.IrodoriTTSEmojiSelector, 
        ]


async def comfy_entrypoint():
    return Extension()

WEB_DIRECTORY = "./web"


