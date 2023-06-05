from flask import Flask, request
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import hashlib
import requests


app = Flask(__name__)
url = 'http://43.143.222.78:9999/interview'

def get_file_hash(file_data):
    file_hash = hashlib.sha256()
    file_hash.update(file_data)
    return file_hash.hexdigest()

@app.route('/', methods=['POST'])
def handle_post():
    data = request.get_data()
    print('Received file hash:', get_file_hash(data))
    with open('temp_audio.wav', 'wb') as f:
        f.write(data)
    result = pipe(audio_in='temp_audio.wav')
    result = result['text']
    print(result)
    data = {
        'prompt': result,
    }
    response = requests.post(url=url, data=data).json()['response'][0]
    return response


if __name__ == '__main__':
    print("开始初始化......")
    pipe = pipeline('auto-speech-recognition', 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                 device='cpu')
    print("模型已加载完成")
    app.run(host='0.0.0.0', port=5000)
