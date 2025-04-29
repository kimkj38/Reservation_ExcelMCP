import requests
import json
import uuid

def chat_with_api():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    # 대화 기록을 저장할 리스트
    conversation_history = []
    # 세션 유지를 위한 사용자 ID
    user_id = str(uuid.uuid4())
    
    print("병원 예약 시스템과 대화를 시작합니다. (종료하려면 'quit' 또는 'exit' 입력)")
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n💬 사용자 입력: ")
        
        # 종료 조건 확인
        if user_input.lower() in ['quit', 'exit']:
            print("대화를 종료합니다.")
            break
        
        # 사용자 메시지 추가
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # API 요청 데이터 구성
        data = {
            "model": "gemma3-tools",
            "user": user_id,  # 세션 유지를 위한 사용자 ID 포함
            "messages": conversation_history
        }
        
        try:
            # API 호출
            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()
            
            # 응답 처리
            if response.status_code == 200 and "choices" in response_data:
                assistant_message = response_data["choices"][0]["message"]
                print(f"\n🤖: {assistant_message['content']}")
                
                # 응답을 대화 기록에 추가
                conversation_history.append(assistant_message)
            else:
                print(f"\n오류 발생: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
                
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    chat_with_api()