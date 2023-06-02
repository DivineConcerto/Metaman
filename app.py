import requests
from flask import Flask, request
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


app = Flask(__name__)
url = 'http://127.0.0.1/interview'

@app.route('/audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return 'No audio file found', 400

    audio_file = request.files['audio']
    result = p(audio_file)
    return submin_to_gpt(result)
    # 在这里进行音频处理逻辑
    # 例如，保存音频文件或进行语音识别等
    # 示例：保存音频文件
    # audio_file.save('received_audio.wav')
    # 示例：进行语音识别
    # 假设你使用的是SpeechRecognition库
    # import speech_recognition as sr
    # recognizer = sr.Recognizer()
    # with sr.AudioFile('received_audio.wav') as source:
    #     audio = recognizer.record(source)
    # result = recognizer.recognize_google(audio)

def submin_to_gpt(text):
    data = {
        'text': text
    }
    response = requests.post(url=url, json=data)
    return response.text

if __name__ == '__main__':
    print("开始初始化......")
    p = pipeline('auto-speech-recognition', 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', device='cpu')
    print("模型已加载完成")
    app.run(host='0.0.0.0', port=5000)
