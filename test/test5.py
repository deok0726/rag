import json

# 주어진 JSON 파일
json_data = '''
{
    "status": {
        "code": "20000",
        "message": "OK"
    },
    "result": {
        "topicSeg": [
            ["노트는 어떻게 생성할 수 있나요?", "두 가지 방법이 있습니다."],
            ["클로바노트 앱에서 추가 버튼을 눌러 녹음을 시작하거나, 스마트폰에 저장해둔 녹음 파일을 불러오면 노트가 생성됩니다.",
             "이렇게 만들어진 노트는 앱뿐만 아니라 PC의 클로바노트 웹사이트에서도 연동되어 확인하실 수 있는데요.",
             "클로바노트 사이트에서는 저장된 녹음파일을 불러오면 노트를 만들 수 있습니다."],
            ["북마크는 어떻게 사용하는 건가요?",
             "클로바노트 앱 화면에서 녹음 중간에 북마크 버튼을 누르면, 아래처럼 표시되어 녹음을 마치고 나서도 필요한 구간을 쉽게 찾을 수 있죠.",
             "평소 녹음을 마치고 나면 분명히 다시 찾아보고 싶은 녹음 구간이 있었을 거예요.",
             "그런 순간을 위해 북마크를 제공하고 있답니다."],
            ["그럼 녹음한 음성은 어떻게 들어볼 수 있나요?",
             "생성된 노트에서 기록된 대화를 선택하면 녹음 음성을 다시 들어볼 수 있답니다.",
             "만약 음성 기록이 잘못된 구간이 있다면 다시 한 번 음성을 들어보고 편집 버튼을 눌러 쉽게 바로잡을 수 있죠."]
        ],
        "span": [
            [0, 1],
            [2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11]
        ],
        "inputTokens": 330
    }
}
'''

# JSON 데이터 파싱
data = json.loads(json_data)

# "topicSeg"에서 내용 추출
topic_seg_contents = data["result"]["topicSeg"]

print(type(topic_seg_contents))
print(topic_seg_contents)
print(topic_seg_contents[0])


topic_seg_cleaned = [', '.join(map(str, topic)) for topic in topic_seg_contents]
print(topic_seg_cleaned)
print(topic_seg_cleaned[0])

# 결과 출력
# for topic in topic_seg_contents:
#     print(topic)