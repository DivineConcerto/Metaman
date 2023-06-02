from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# 我的本子显卡是MX450，太垃圾了以至于如果不设置设备为cpu完全不能运行。如果您的显卡比我的强那么请删掉device = 'cpu'！
p = pipeline('auto-speech-recognition', 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',device='cpu')
result = p(audio_in='https://s17.aconvert.com/convert/p3r68-cdx67/n20rs-ehe1t.wav')
print(result)
