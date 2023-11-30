# Machine-Generated-Code-Detection-Demo_Fast-detectGPT
Machine-Generated Code Detection Demo (Fast-detectGPT)

프로그램 소개
사용자에게 코드를 입력 받으면 Code Llama 7B 모델을 이용해 Fast-DetectGPT algorithm을 적용하고, 이를 통해 human-written 또는 machine-generated를 판별하는 과정을 사용자들이 쉽게 체험해 볼 수 있도록 개발된 챗봇 프로그램이다. Python 기반의 웹 프레임워크인 FastAPI를 이용해 제작되었으며, Asynchronous Server Gateway Interface (ASGI) 서버인 Uvicorn를 사용했다. 프로그램의 개발을 위해 HTML, CSS, JavaScript, Python 등의 언어가 사용되었다. <br>

주요 기능
- 사용자는 입력창에 human-written 또는 machine-generated code를 입력한다.<br>
- 주어진 코드에 fast-detectGPT algorithm을 적용해 human-written 또는 machine-generated  code를 판별한다.<br>
- 판별한 결과를 보여준다.<br>

사용 방법
1. Demo - 실제 데모 프로그램을 실행해볼 수 있다. <br>
2. 코드를 입력하고 Send 버튼을 누르면 판별 결과가 생성된다. <br>
3. 입력된 코드가 human-written code인지, 또는 machine-generated code인지 판별한 결과가 출력된다.<br>
4. 반복적으로 질문을 입력하여 테스트 해볼 수 있다. <br>


1) git clone https://github.com/Jhj9/Machine-Generated-Code-Detection-Demo_Fast-detectGPT.git<br>

2) pip install -r requirements.txt<br>

4) uvicorn main:app<br>

5) http://127.0.0.1:8000/ 웹페이지 실행
