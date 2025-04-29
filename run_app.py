import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from utils import ainvoke_graph, astream_graph
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, List, Dict, Any, Annotated, Literal, Optional
import operator
import re

load_dotenv()

# 기본 모델 설정
# model = OpenAI(
#         base_url="http://localhost:8000/v1",
#         api_key="token",  # vLLM은 API 키 필요 없음
#         timeout=600  # 10분 타임아웃
#     )
model = ChatOllama(model="PetrosStav/gemma3-tools:27b", num_ctx=20000, num_predict=1024)
# model = ChatOllama(model="llama3.1:8b", num_ctx=20000, num_predict=1024)
# model = ChatOllama(model="qwen2.5:14b", num_ctx=20000, num_predict=1024)
# model = ChatOpenAI(model="gpt-4o-mini")

# 각 에이전트별 시스템 프롬프트 정의

# MCP 설정 관련 함수
def load_mcp_config():
    try:
        with open("./mcp_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return None

def create_server_config():
    config = load_mcp_config()
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []),
                    "transport": "stdio",
                }
            elif "url" in server_config_data:
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse",
                }

    return server_config


class AgentState(TypedDict):
        messages: str
        # booking_info: Dict[str, str]
        missing_info: List
        excel_info: str
        available_row: Optional[int]
        generation: str
        formatting: Dict[str, str]
        client: Any
        row: str

def extract_missing_info(state: AgentState):

    prompt = PromptTemplate(
            template="""당신은 사용자가 제공한 정보를 받아 병원 진료 예약을 위해 더 필요한 정보를 추출하는 유능한 AI Agent 입니다.\n 

            제공되는 정보는 다음과 같습니다.\n
            {messages}

            Rules
            1. 필수 항목은 '환자명','생년월일','예약일', '예약시간', '연락처' 입니다.
            2. 제공된 정보 중 각 항목이 포함되어 있는지 확인 후 내용이 없는 항목들을 나열하여 출력하세요.
            3. 모든 항목이 제공되었다면 빈 문자열 ""을 출력하세요. 설명은 하지마세요. 
            4. 부연설명은 절대 하지 마세요. 

            """,
            input_variables=["messages"],
    )

    # 에이전트 생성 및 호출
    messages = state['messages']
    agent = prompt | model | StrOutputParser()
    
    response = agent.invoke({"messages": messages})
    # print("@@@", state['messages'])
    print("누락된 필수정보:", response)


    missing = []

    required_fields = ['환자명', '생년월일', '예약일', '예약시간', '연락처']
    for x in required_fields:
        if x in response:
            missing.append(x)

    return {"messages": messages, "missing_info": missing}


def judge(state: AgentState):

    missing_info = state['missing_info']

    if len(missing_info)>0:
        return "incomplete"
    else:
        return "complete"


def request_info(state: AgentState):
    last_word = state['missing_info'][-1]
    info = ', '.join(state['missing_info'])

    if last_word in ['환자명', '생년월일', '예약시간']:
       generation = f"진료예약을 위해서는 {info}이 필요합니다. 추가 정보를 말씀해주시면 예약 진행을 도와드리겠습니다."
    else:
        generation = f"진료예약을 위해서는 {info}가 필요합니다. 추가 정보를 말씀해주시면 예약 진행을 도와드리겠습니다."
    

    return {"messages": state['messages'], "missing_info": state['missing_info'], "generation": generation}


async def read_excel(state: AgentState, client):

    print("------예약 가능 여부를 조회 중입니다------")

    prompt = """당신은 'read_sheet_data' 툴을 활용하여 엑셀 내용을 읽어오는 AI Agent입니다.
            아래 경로의 엑셀 파일에서 예약 정보를 읽어와 주세요.

            파일 정보:
            fileAbsolutePath: "C:/Users/kyongjun/Desktop/hospital2.xlsx"
            sheetName: "Sheet1"
            range: "A1:F100"

            이 정보를 사용하여 read_sheet_data 툴을 호출하고 아래와 같은 JSON 형식으로 엑셀 내용와 비어있는 가장 윗 행의 번호를 출력하세요.
            {{"contents": "엑셀 정보", "row": "비어있는 가장 윗 행의 번호"}}
            부가설명은 하지마세요."""
  
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="엑셀 파일의 내용을 읽어와주세요.")
    ]
    
    tools = client.get_tools()
    agent = create_react_agent(model=model, tools=tools, prompt=prompt)
    
    response = await agent.ainvoke({"messages": messages})
    
    # 응답에서 마지막 메시지의 내용을 추출
    if isinstance(response, dict) and "messages" in response:
        last_message = response["messages"][-1]
        content = last_message.content if hasattr(last_message, 'content') else last_message.get("content", "")
        
        # 만약 content가 이미 딕셔너리라면 그대로 사용하고, 문자열이라면 JSON으로 파싱
        if isinstance(content, dict):
            content_dict = content
        else:
            # JSON 문자열에서 JSON 객체 부분만 추출해서 파싱
            try:
                import re
                import json
                # JSON 형식의 문자열을 찾기
                json_matches = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_matches:
                    json_str = json_matches.group(1)
                    content_dict = json.loads(json_str)
                else:
                    # JSON 문자열이 없으면 전체를 파싱 시도
                    content_dict = json.loads(content)
            except json.JSONDecodeError:
                return {"messages": state["messages"], "excel_info": "JSON 파싱 실패", "generation": "JSON 파싱 실패"}
        
        return {"messages": state["messages"], "excel_info": content_dict['contents'], "row": content_dict['row']}
    
    return {"messages": state["messages"], "excel_info": "엑셀 파일을 읽는데 실패했습니다.", "generation": "엑셀 파일을 읽는데 실패했습니다."}


def check_duplicate(state: AgentState):
    

    prompt = PromptTemplate(
            template="""당신은 고객이 요청한 예약정보와 기존 예약 기록들을 비교하여 중복 여부를 확인하는 AI Agent입니다.\n
            선예약한 고객이 있다면 해당 시간에 예약을 잡으면 안되므로 매우 중요한 일입니다.

            고객이 요청한 정보는 다음과 같습니다.\n
            {messages} \n
            기존 예약기록들은 다음과 같습니다. \n
            {excel_info}

            다음 프로세스에 따라 추론하세요.
            1. 고객이 요청한 정보에서 예약일자와 예약시간을 확인
            2. 만약 고객의 정보 중 동일 항목에 대해 2가지 이상의 정보가 주어진다면 더 최신 정보인 아래쪽 정보를 활용하세요.
            3. 기존 예약기록들로부터 예약일자 및 시간 확인
            4. 고객이 요청한 시간대와 겹치는 기존 예약이 있는지 비교
            5. 겹치는 기존 예약이 있다면 "yes"를 출력하고, 그렇지 않다면 "no"를 출력. 근거를 1줄로 설명하세요.
            """,
            input_variables=["messages","excel_info"],
    )

    # 에이전트 생성 및 호출
    messages = state['messages']
    excel_info = state['excel_info']
    agent = prompt | model | StrOutputParser()
    
    response = agent.invoke({"messages": messages, "excel_info": excel_info})

    print("고객 요청:", messages, "\n")
    print("엑셀 내용:", excel_info, "\n")
    print("중복 여부:", response, "\n")

    if "yes" in response:
        return "yes"
    else:
        return "no"


async def propose_candidate(state: AgentState, client):

    messages = state['messages']
    excel_info = state['excel_info']
    prompt = f"당신은 선예약이 있는 시간에 예약을 요청한 고객에게 가능한 시간대를 제시해주는 친절한 AI Agent입니다.\n \
                고객이 요청한 정보는 다음과 같습니다.\n \
                {messages} \n \
                기존 예약기록들은 다음과 같습니다. \n \
                {excel_info} \n \n\
                다음 프로세스에 따라 추론하세요. \n \
                1. 고객이 요청한 정보에서 예약일자와 예약시간을 확인 \n\
                2. 엑셀파일에 기록된 예약기록들로부터 이미 예약된 일자 및 시간을 확인 \n\
                3. 기존 예약들과 겹치지 않으면서 고객이 요청한 예약과 유사한 시간대를 제시."
    
    inputs = [
        SystemMessage(content=prompt),
    ]
    
    tools = client.get_tools()
    agent = create_react_agent(model=model, tools=tools, prompt=prompt)
    
    response = await agent.ainvoke({"messages": inputs})
    
    # 응답에서 마지막 메시지의 내용을 추출
    if isinstance(response, dict) and "messages" in response:
        last_message = response["messages"][-1]
        content = last_message.content if hasattr(last_message, 'content') else last_message.get("content", "")

        return {"messages": state["messages"], "excel_info": excel_info, "generation": content}


def get_format(state: AgentState):

    prompt = PromptTemplate(
            template="""당신은 고객이 요청한 정보를 정해진 양식에 맞게 재구성하는 유능한 AI Agent입니다.\n
            고객이 요청한 정보는 다음과 같습니다. 아래쪽의 정보가 더 최신정보입니다.
            {messages} 
            위 정보로부터 '환자명', '생년월일', '예약일', '예약시간', '연락처', '비고'를 추출하여 JSON 포맷으로 출력하세요.\n
            


            Example: 
            {{"환자명": "최민호", "생년월일": "1994-05-16", "예약일": "2025-04-19", "예약시간": "15:00", "연락처": "010-5464-8977", "비고": ""}}    
            
            Rules
            1. JSON 양식을 꼭 준수해야 합니다. 이는 매우 중요한 규칙입니다.
            2. 부가설명은 하지 마세요.
            3. 예시에서 보여준 양식을 준수하세요.
            4. 특정 항목의 정보가 없는 경우 예시의 '비고'와 같이 빈 문자열로 남겨주세요.
            5. 같은 항목에 대해 2개 이상의 정보가 있다면 더 최신 정보인 아래쪽 정보를를 반영하여 작성하세요.
            
            """,
            input_variables=["messages"],
    )

    # 에이전트 생성 및 호출
    messages = state['messages']

    agent = prompt | model | JsonOutputParser()
    
    response = agent.invoke({"messages": messages})

    new_format = {"data": [[]], "fileAbsolutePath":  "C:/Users/kyongjun/Desktop/hospital2.xlsx", "range": f"A{state['row']}:F{state['row']}", "sheetName": "Sheet1"}
    for vals in response.values():
        new_format['data'][0].append(vals)
    

    # print("양식 설정 결과:", str(new_format))

    return {"messages": messages, "formatting": str(new_format)}



async def write_excel(state: AgentState, client):

    print("------예약을 진행 중입니다------")

    prompt = ""
  
    messages = [
        HumanMessage(content="'write_sheet_data' 툴을 활용하여 다음 정보를 엑셀에 작성하세요."),
        HumanMessage(content=state['formatting'])
    ]
    
    tools = client.get_tools()
    agent = create_react_agent(model=model, tools=tools, prompt=prompt)
    
    response = await agent.ainvoke({"messages": messages})
    
    # 응답에서 마지막 메시지의 내용을 추출
    if isinstance(response, dict) and "messages" in response:
        last_message = response["messages"][-1]
        content = last_message.content if hasattr(last_message, 'content') else last_message.get("content", "")
        # print("엑셀 작성 결과:", content)

        return {"messages": state["messages"], "excel_info": content, "generation": "예약이 완료되었습니다. 좋은 하루 되세요!"}
    
    return {"messages": state["messages"], "generation": "예약이 완료되었습니다. 좋은 하루 되세요!"}




# 그래프 구성
async def build_agent_graph():

    server_config = create_server_config()
    
    client = MultiServerMCPClient(server_config)

    await client.__aenter__()

    # 초기 상태에 client 포함
    initial_state = {"messages": "", "missing_info": [], "excel_info": "", "available_row": None, "generation": "", "client": client}

    async def read_excel_wrapper(state):
        return await read_excel(state, client)
    
    async def propose_candidate_wrapper(state):
        return await propose_candidate(state, client)

    async def write_excel_wrapper(state):
        return await write_excel(state, client)

    
    # 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가 - 비동기 래퍼 함수 사용
    workflow.add_node("extract_missing_info", extract_missing_info)
    # workflow.add_node("judge", judge_node)
    workflow.add_node("read_excel", read_excel_wrapper)
    workflow.add_node("request_info", request_info)
    workflow.add_node("propose_candidate", propose_candidate_wrapper)
    workflow.add_node("get_format", get_format)
    workflow.add_node("write_excel", write_excel_wrapper)
    
    # 엣지 추가
    workflow.add_edge(START, "extract_missing_info")
    workflow.add_conditional_edges("extract_missing_info",
                    judge,
                    {
                        "complete": "read_excel",
                        "incomplete": "request_info",
                    },
                    )
    workflow.add_edge("request_info", END)
    workflow.add_conditional_edges("read_excel",
                        check_duplicate,
                        {"yes": "propose_candidate",
                        "no": "get_format"}
                        )
    workflow.add_edge("propose_candidate", END)
    workflow.add_edge("get_format", "write_excel")
    workflow.add_edge("write_excel", END)

    return workflow.compile(), initial_state 


async def interactive_chat():
    graph, initial_state = await build_agent_graph()
    
    print("병원 예약 도우미와 대화를 시작합니다. (종료하려면 'exit' 입력)")
    messages = ""
    
    # initial_state에서 state 초기화
    state = initial_state.copy()

    while True:

        query = input("\n💬 사용자 입력: ")
        
        if query.lower() in ["exit", "quit"]:
            print("대화를 종료합니다.")
            break
            
        try:
            # 그래프 실행
            messages += f" {query}"
            state["messages"] = messages
            
            # 상태를 직접 전달
            response = await graph.ainvoke(state)
            # print("@@@", response)
            print(f"\n🤖 {response['generation']}")
            
            # 상태 업데이트
            state = response
            messages = response['messages']
            
            # 예약 완료 시 메시지 초기화
            if response['generation'] == "예약이 완료되었습니다. 좋은 하루 되세요!":
                messages = ""
                
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(interactive_chat())



